#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "index.h"

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

template<class T>
void shuffle(T *arr, int size)
{
    // Fisher Yates Shuffle
    // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    for (int i = size - 1; i > 0; i--) 
    { 
        int j = rand() % (i + 1); 
        T tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

int * buildIndices(int dataSize, int maxIter)
{
    int *indices = (int*)malloc(maxIter * dataSize * sizeof(int));
    for (int iter = 0; iter < maxIter; iter++)
    {
        for (int idx = 0; idx < dataSize; idx++)
        {
            indices[iter * dataSize + idx] = idx;
        }
        shuffle(&indices[iter * dataSize], dataSize);
    }
    return indices;
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

struct LinearModel findBest(int trainSize, struct LinearModel *candidates, int candidatesSize, double2 *data)
{
    double3 cs[candidatesSize];

    // reduce
    for (int t = 0; t < candidatesSize; t++)
    {
        struct LinearModel m = candidates[t];
        struct dist_functor dist_op = dist_functor(m.slope, m.intercept);
        thrust::plus<double> add_op;
        double sum = thrust::transform_reduce(data, data + trainSize, dist_op, 0.0, add_op);
        // mean squarred error
        cs[t].x = m.slope;
        cs[t].y = m.intercept;
        cs[t].z = sum / trainSize;
    }

    struct min_model min_op;
    double3 bestM = thrust::reduce(cs, cs + candidatesSize, getInfModel(), min_op);
    struct LinearModel m;
    m.slope = bestM.x;
    m.intercept = bestM.y;
    m.error = bestM.z;
    return m;
}

struct LinearModel singleIter(int iter, double2 *rawData, int dataSize, double thresh, int trainSize, int wellCount)
{
    // create a copy of the data (maybe not needed in the future)
    double2 data[dataSize];
    for (int i = 0; i < dataSize; i++)
    {
        data[i] = rawData[i];
    }
    shuffle(data, dataSize);
    double2 *d_data;
    cudaMalloc(&d_data, dataSize * sizeof(double2));
    cudaMemcpy(d_data, data, dataSize * sizeof(double2), cudaMemcpyHostToDevice);

    // the number of unique combinations of all data points
    int numCombinations = triangMax(trainSize);

    int threadsPerBlock = 32;
    int numBlocks = numCombinations / threadsPerBlock;
    // ignore some combinations OR fill them up with duplicates
    numCombinations = numBlocks * threadsPerBlock;

    // produce every possible model
    int candidateSize = numCombinations * sizeof(struct LinearModel);
    struct LinearModel *d_candidateModels;
    cudaMalloc(&d_candidateModels, candidateSize);
    
    buildModel<<<numBlocks, threadsPerBlock>>>(d_data, numCombinations, d_candidateModels);
    struct LinearModel *candidateModels = (struct LinearModel*)malloc(candidateSize);
    cudaMemcpy(candidateModels, d_candidateModels, candidateSize, cudaMemcpyDeviceToHost);
    cudaFree(d_candidateModels);

    struct LinearModel bestModel = findBest(trainSize, candidateModels, numCombinations, data);

    free(candidateModels);

    // evaluate the models
    int numGood;
    int tmpInlierIndices[dataSize - trainSize];
    numGood = checkGood(bestModel, data, dataSize, trainSize, thresh, tmpInlierIndices);
    if (numGood < wellCount)
    {
        struct LinearModel fail;
        fail.slope = 0;
        fail.intercept = 0;
        fail.error = std::numeric_limits<double>::infinity();
        return fail;
    }
    else
    {
        // reorder (put the new inliers first)
        for (int g = 0; g < numGood; g++)
        {
            double2 tmp = data[trainSize + g];
            data[trainSize + g] = data[tmpInlierIndices[g]];
            data[tmpInlierIndices[g]] = tmp;
        }
    }

    int numTrainAndGood = trainSize + numGood;
    int numInlierComb = triangMax(numTrainAndGood);

    // fit again using the new inlier indices
    numBlocks = numInlierComb / threadsPerBlock;
    numInlierComb = numBlocks * threadsPerBlock;
    struct LinearModel *inlierModels = (struct LinearModel*)malloc(numInlierComb * sizeof(struct LinearModel));
    struct LinearModel *d_inlierModels;
    cudaMalloc(&d_inlierModels, numInlierComb * sizeof(inlierModels[0]));

    buildModel<<<numBlocks, threadsPerBlock>>>(d_data, numInlierComb, d_inlierModels);

    cudaMemcpy(inlierModels, d_inlierModels, numInlierComb * sizeof(inlierModels[0]), cudaMemcpyDeviceToHost);
    cudaFree(d_inlierModels);

    bestModel = findBest(numTrainAndGood, inlierModels, numInlierComb, data);

    free(inlierModels);

    cudaFree(d_data);

    return bestModel;
}

struct LinearModel ransac(double2 *data, int dataSize, int maxIter, double thresh, int trainSize, int wellCount)
{
    struct LinearModel best;
    best.error = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < maxIter; iter++)
    {
        struct LinearModel m = singleIter(iter, data, dataSize, thresh, trainSize, wellCount);
        std::cout << iter << " " << m.slope << " " << m.intercept << " " << m.error << std::endl;
        best = minModel(m, best);
    }
    std::cout << "error: " << best.error << std::endl;

    if (best.error == std::numeric_limits<double>::infinity())
    {
        std::cerr << "RANSAC failed" << std::endl;
    }
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
    const int numIters = 20;
    float trainRatio = .3f;
    float wellRatio = .1f;
    double errorThresh = .3;
    struct LinearModel m = ransac(&data[0], dataSize, numIters, errorThresh, (int)(trainRatio * dataSize), (int)(wellRatio * dataSize));
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
