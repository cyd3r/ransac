#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#ifndef USE_GPU
#define USE_GPU 1
#endif

// #include "fit.cu"

struct LinearModel
{
    double slope;
    double intercept;
    double numInliers;
};

struct LinearModel maxModel(struct LinearModel a, struct LinearModel b)
{
    return a.numInliers > b.numInliers ? a : b;
}

struct thresh_op
{
    double slope, intercept, thresh;
    thresh_op(const double _slope, const double _intercept, const double _thresh)
    {
        slope = _slope;
        intercept = _intercept;
        thresh = _thresh;
    }

    __host__ __device__
    int operator()(const double2 point) const
    {
        return std::abs(slope * point.x - point.y + intercept) < thresh ? 1 : 0;
    }
};

int countInliers(struct LinearModel model, double2 *data, int dataSize, double thresh, int *good)
{
    thresh *= std::sqrt(model.slope * model.slope + 1);
#if USE_GPU
    thrust::device_vector<double2> d_data(data, data + dataSize);
    thrust::device_vector<int> isGood(dataSize);
    // calculate distances and compare them to the threshold
    thrust::transform(thrust::device, d_data.begin(), d_data.end(), isGood.begin(), thresh_op(model.slope, model.intercept, thresh));

    thrust::device_vector<int> d_indices(dataSize);
    thrust::inclusive_scan(thrust::device, isGood.begin(), isGood.end(), d_indices.begin());

    thrust::host_vector<int> indices(d_indices.begin(), d_indices.end());
#else
    thrust::host_vector<int> isGood(dataSize);
    thrust::transform(data, data + dataSize, isGood.begin(), thresh_op(model.slope, model.intercept, thresh));
    thrust::host_vector<int> indices(dataSize);
    thrust::inclusive_scan(isGood.begin(), isGood.end(), indices.begin());
#endif

    for (int i = dataSize - 1; i >= 0; i--)
    {
        //     x     x x x
        // 0 0 1 1 1 2 3 4 4 4
        // 0 1 2 3 4 5 6 7 8 9
        // 2 5 6 7
        good[indices[i]] = i;
    }
    int goodSize = indices[dataSize - 1];

    return goodSize;
}

struct LinearModel singleIter(int iter, double2 *data, int dataSize, double thresh, int wellCount, int *inliers, int *inliersSize)
{
    if (iter >= 2 * dataSize)
    {
        std::cerr << "Dataset is too small" << std::endl;
        exit(1);
    }

    inliers[0] = 2 * iter;
    inliers[1] = 2 * iter + 1;

    double2 pI = data[inliers[0]];
    double2 pK = data[inliers[1]];
    struct LinearModel bestModel;
    bestModel.slope = (pK.y - pI.y) / (pK.x - pI.x);
    bestModel.intercept = pI.y - bestModel.slope * pI.x;

    // evaluate the models
    int numGood;
    numGood = countInliers(bestModel, data, dataSize, thresh, inliers + 2);
    std::cout << inliers[0] << " " << inliers[1] << " " << numGood << std::endl;
    *inliersSize = numGood + 2;
    if (numGood < wellCount)
    {
        std::cout << "num good: " << numGood << "<" << wellCount << std::endl;
        struct LinearModel fail;
        fail.slope = std::numeric_limits<double>::quiet_NaN();
        fail.intercept = std::numeric_limits<double>::quiet_NaN();
        fail.numInliers = -1;
        return fail;
    }

    bestModel.numInliers = numGood;

    return bestModel;
}

struct LinearModel ransac(double2 *data, int dataSize, int maxIter, double thresh, int wellCount)
{
    struct LinearModel best;
    int *inliers = (int *)malloc(dataSize * sizeof(int));
    int *bestInliers = (int *)malloc(dataSize * sizeof(int));
    int bestInliersSize = 0;
    int inliersSize;
    best.numInliers = -1;
    int bestIter = -1;
    for (int iter = 0; iter < maxIter; iter++)
    {
        struct LinearModel m = singleIter(iter, data, dataSize, thresh, wellCount, inliers, &inliersSize);
        std::cout << iter << " " << m.slope << " " << m.intercept << " " << m.numInliers << std::endl;
        if (m.numInliers > best.numInliers)
        {
            bestIter = iter;
            memcpy(bestInliers, inliers, inliersSize * sizeof(int));
            bestInliersSize = inliersSize;
        }
        best = maxModel(m, best);
    }

    if (bestIter < 0)
    {
        std::cerr << "RANSAC failed" << std::endl;
    }
    else
    {
        std::cout << "best iter: " << bestIter << std::endl;
        // write the inliers to a file
        FILE *f = fopen("inliers.txt", "w");
        for (int i = 0; i < bestInliersSize; i++)
        {
            fprintf(f, "%d\n", bestInliers[i]);
        }
        fclose(f);
        // std::cout << "fit..." << std::endl;
        // fit(data, bestInliers, bestInliersSize, best.slope, best.intercept);
    }

    free(inliers);
    free(bestInliers);

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
    const int numIters = 7;
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
    fprintf(f, "%f %f %f\n", m.slope, m.intercept, m.numInliers);
    fclose(f);
    return 0;
}
