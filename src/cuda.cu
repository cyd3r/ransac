#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "index.h"

void printModel(double slope, double intercept, double error)
{
    std::cout << "slope " << slope << " intercept " << intercept << " " << error << std::endl;
}

double3 getInfModel()
{
    double3 m;
    m.z = std::numeric_limits<double>::infinity();
    return m;
}

struct dist_functor {
    double slope = 0;
    double intercept = 0;
    dist_functor(const double _slope, const double _intercept)
    {
        slope = _slope;
        intercept = _intercept;
    }

    __host__ __device__
    double operator()(const double2 &p) const
    {
        double dist = std::abs(slope * p.x - p.y + intercept) / std::sqrt(slope * slope + 1);
        return dist * dist;
    }
};

struct min_model
{
    __host__ __device__
    double3 operator()(const double3 &a, const double3 &b) const
    {
        return a.z < b.z ? a : b;
    }
};

struct LinearModel
{
    double slope;
    double intercept;
    double error;
};

struct LinearModel minModel(struct LinearModel a, struct LinearModel b)
{
    return a.error < b.error ? a : b;
}

double distance(struct LinearModel model, double2 p)
{
    return std::abs(model.slope * p.x - p.y + model.intercept) / std::sqrt(model.slope * model.slope + 1);
}

__device__ void triangToTuple(int t, int &x, int &y)
{
    y = (int)((1 + (int)sqrtf(1 + 8 * t)) / 2);
    x = t - (y * (y - 1)) / 2;
}

__global__ void buildModel(double2 *data, int numCombinations, struct LinearModel *models)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i, k;
    triangToTuple(t, i, k);

    double2 pI = data[i];
    double2 pK = data[k];

    struct LinearModel m;
    m.slope = (pK.y - pI.y) / (pK.x - pI.x);
    m.intercept = pI.y - m.slope * pI.x;

    models[t] = m;
}

int checkGood(struct LinearModel m, double2 *data, int dataSize, int trainSize, double thresh, int *good)
{
    // reduce
    int goodSize = 0;
    for (int d = trainSize; d < dataSize; d++)
    {
        double2 p = data[d];
        double dist = distance(m, p);
        if (dist < thresh)
        {
            good[goodSize] = d;
            goodSize++;
        }
    }

    return goodSize;
}

struct LinearModel findBest(int trainSize, struct LinearModel *candidates, int candidatesSize, double2 *data, double2 *d_data)
{
    std::cout << candidatesSize * sizeof(double2) / 1024.0 / 1024.0 / 1024.0 << " GB" << std::endl;
    // double3 cs[candidatesSize];
    double3 *cs = (double3 *)malloc(candidatesSize * sizeof(double3));

    // reduce
    for (int t = 0; t < candidatesSize; t++)
    {
        struct LinearModel m = candidates[t];
        struct dist_functor dist_op = dist_functor(m.slope, m.intercept);
        double sum = thrust::transform_reduce(data, data + trainSize, dist_op, 0.0, thrust::plus<double>());
        // double sum = thrust::transform_reduce(thrust::device, d_data, d_data + trainSize, dist_op, 0.0, thrust::plus<double>());
        // mean squarred error
        cs[t].x = m.slope;
        cs[t].y = m.intercept;
        cs[t].z = sum / trainSize;
    }

    std::cout << "end reduce" << std::endl;

    struct min_model min_op;
    double3 bestM = thrust::reduce(cs, cs + candidatesSize, getInfModel(), min_op);
    free(cs);
    struct LinearModel m;
    m.slope = bestM.x;
    m.intercept = bestM.y;
    m.error = bestM.z;
    return m;
}

struct LinearModel singleIter(int iter, double2 *rawData, int dataSize, double thresh, int wellCount, int *inliers, int *inliersSize)
{
    if (iter >= 2 * dataSize)
    {
        std::cerr << "Dataset is too small" << std::endl;
        exit(1);
    }

    double2 data[dataSize];
    memcpy(data, rawData, dataSize * sizeof(double2));

    inliers[0] = 2 * iter;
    inliers[1] = 2 * iter + 1;

    double2 *d_data;
    cudaMalloc(&d_data, dataSize * sizeof(double2));
    cudaMemcpy(d_data, data, dataSize * sizeof(double2), cudaMemcpyHostToDevice);

    double2 pI = data[inliers[0]];
    double2 pK = data[inliers[1]];
    struct LinearModel bestModel;
    bestModel.slope = (pK.y - pI.y) / (pK.x - pI.x);
    bestModel.intercept = pI.y - bestModel.slope * pI.x;

    // evaluate the models
    int numGood;
    numGood = checkGood(bestModel, data, dataSize, 2, thresh, inliers + 2);
    *inliersSize = numGood + 2;
    if (numGood < wellCount)
    {
        std::cout << "num good: " << numGood << "<" << wellCount << std::endl;
        struct LinearModel fail;
        fail.slope = std::numeric_limits<double>::quiet_NaN();
        fail.intercept = std::numeric_limits<double>::quiet_NaN();
        fail.error = std::numeric_limits<double>::infinity();
        return fail;
    }

    bestModel.error = numGood;

    cudaFree(d_data);

    return bestModel;
}

struct LinearModel fit(double2 *rawData, int *inliers, int inliersSize)
{
    std::cout << inliersSize << " inliers, " << triangMax(inliersSize) << " combinations" << std::endl;
    double2 data[inliersSize];
    memcpy(data, rawData, inliersSize * sizeof(double2));

    double2 *d_data;
    cudaMalloc(&d_data, inliersSize * sizeof(double2));
    cudaMemcpy(d_data, data, inliersSize * sizeof(double2), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numInlierComb = triangMax(inliersSize);

    int numBlocks = numInlierComb / threadsPerBlock;
    numInlierComb = numBlocks * threadsPerBlock;

    struct LinearModel *inlierModels = (struct LinearModel*)malloc(numInlierComb * sizeof(struct LinearModel));
    struct LinearModel *d_inlierModels;
    cudaMalloc(&d_inlierModels, numInlierComb * sizeof(inlierModels[0]));

    buildModel<<<numBlocks, threadsPerBlock>>>(d_data, numInlierComb, d_inlierModels);

    cudaMemcpy(inlierModels, d_inlierModels, numInlierComb * sizeof(inlierModels[0]), cudaMemcpyDeviceToHost);
    cudaFree(d_inlierModels);

    struct LinearModel bestModel = findBest(inliersSize, inlierModels, numInlierComb, data, d_data);

    free(inlierModels);
    cudaFree(d_data);

    return bestModel;
}

struct LinearModel ransac(double2 *data, int dataSize, int maxIter, double thresh, int wellCount)
{
    struct LinearModel best;
    int *inliers = (int *)malloc(dataSize * sizeof(int));
    int inliersSize;
    best.error = std::numeric_limits<double>::infinity();
    int bestIter = -1;
    for (int iter = 0; iter < maxIter; iter++)
    {
        struct LinearModel m = singleIter(iter, data, dataSize, thresh, wellCount, inliers, &inliersSize);
        std::cout << iter << " " << m.slope << " " << m.intercept << " " << m.error << std::endl;
        if (m.error < best.error)
        {
            bestIter = iter;
        }
        best = minModel(m, best);
    }

    if (bestIter < 0)
    {
        std::cerr << "RANSAC failed" << std::endl;
    }
    else
    {
        // write the inliers to a file
        FILE *f = fopen("inliers.txt", "w");
        for (int i = 0; i < inliersSize; i++)
        {
            fprintf(f, "%d\n", inliers[i]);
        }
        fclose(f);
        // best = fit(data, inliers, inliersSize);
    }

    free(inliers);

    return best;
}

std::vector<double2> readCSV(const char *path)
{
    std::ifstream file(path);
    std::vector<double2> data;

    std::string line;
    while (getline(file, line))
    {
        double2 p;
        sscanf(line.c_str(), "%lf,%lf", &p.x, &p.y);
        data.push_back(p);
    }
    file.close();

    return data;
}

int main(int argc, char const *argv[])
{
    // srand(time(NULL));
    srand(420);
    std::vector<double2> data = readCSV("points.csv");
    int dataSize = data.size();

    clock_t t0 = clock();
    const int numIters = 25;
    float wellRatio = .01f;
    double errorThresh = .4;
    struct LinearModel m = ransac(&data[0], dataSize, numIters, errorThresh, (int)(wellRatio * dataSize));
    clock_t t1 = clock();
    double elapsed_secs = double(t1 - t0) / CLOCKS_PER_SEC;

    std::cout << "Best Model (slope, intercept): " << m.slope << " " << m.intercept << std::endl;
    std::cout << "Time taken: " << elapsed_secs << "s" << std::endl;

    // write the results to a file for visualisation
    FILE *f = fopen("results.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    fprintf(f, "%f %f %f\n", m.slope, m.intercept, m.error);
    fclose(f);
    return 0;
}
