#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "index.h"

struct Point
{
    double x;
    double y;
};

struct LinearModel
{
    double slope;
    double intercept;
};

double distance(struct LinearModel model, struct Point p)
{
    return std::abs(model.slope * p.x - p.y + model.intercept) / std::sqrt(model.slope * model.slope + 1);
}

void shuffle(int *arr, int size)
{
    // Fisher Yates Shuffle
    // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    for (int i = size - 1; i > 0; i--) 
    { 
        int j = rand() % (i + 1); 
        int tmp = arr[i];
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

__global__ void buildModel(double *data, int dataSize, int *indices, int iter, int numCombinations, struct LinearModel *models)
{
    // int t = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    int i, k;
    triangToTuple(t, i, k);

    double *pI = &data[2 * indices[iter * dataSize + i]];
    double *pK = &data[2 * indices[iter * dataSize + k]];

    struct LinearModel m;
    m.slope = (pK[1] - pI[1]) / (pK[0] - pI[0]);
    m.intercept = pI[1] - m.slope * pI[0];

    models[iter * numCombinations + t] = m;
}

int checkGood(struct LinearModel *bestIter, double *data, int dataSize, int *indices, int trainSize, double thresh, int iter, int *good)
{
    struct LinearModel m = bestIter[iter];
    // reduce
    // int good[dataSize - trainSize];
    int goodSize = 0;
    for (int d = trainSize; d < dataSize; d++)
    {
        struct Point p;
        p.x = data[2 * indices[iter * dataSize + d]];
        p.y = data[2 * indices[iter * dataSize + d] + 1];
        // double dist = distance(m, data[indices[iter * dataSize + d]]);
        double dist = distance(m, p);
        if (dist < thresh)
        {
            good[goodSize] = d;
            goodSize++;
        }
    }

    return goodSize;
}

struct LinearModel ransac(double *data, int dataSize, int maxIter, double thresh, int trainSize, int wellCount)
{
    int *indices = buildIndices(dataSize, maxIter);
    int indicesSize = dataSize * maxIter * sizeof(int);

    int *d_indices;
    cudaMalloc(&d_indices, indicesSize);
    cudaMemcpy(d_indices, indices, indicesSize, cudaMemcpyHostToDevice);

    double *d_data;
    cudaMalloc(&d_data, 2 * dataSize * sizeof(double));
    cudaMemcpy(d_data, data, 2 * dataSize * sizeof(double), cudaMemcpyHostToDevice);

    // the number of unique combinations of all data points
    int numCombinations = triangMax(trainSize);

    // produce every possible model
    int candidateSize = maxIter * numCombinations * sizeof(struct LinearModel);
    struct LinearModel *d_candidateModels;
    cudaMalloc(&d_candidateModels, candidateSize);
    
    std::cout << "num comb " << numCombinations << std::endl;
    for (int iter = 0; iter < maxIter; iter++)
    {
        buildModel<<<1, numCombinations>>>(d_data, dataSize, d_indices, iter, numCombinations, d_candidateModels);
    }
    struct LinearModel *candidateModels = (struct LinearModel*)malloc(candidateSize);
    cudaMemcpy(candidateModels, d_candidateModels, candidateSize, cudaMemcpyDeviceToHost);
    cudaFree(d_candidateModels);

    free(candidateModels);

    // find the best model for each iteration
    struct LinearModel bestCandidateModels[maxIter];
    for (int iter = 0; iter < maxIter; iter++)
    {
        double minError = std::numeric_limits<double>::infinity();
        struct LinearModel bestModel;
        // reduce
        for (int t = 0; t < numCombinations; t++)
        {
            int i, k;
            tuple_triang(t, i, k);
            struct LinearModel m = candidateModels[iter * numCombinations + t];
            double sum = 0;
            // reduce
            for (int d = 0; d < trainSize; d++)
            {
                struct Point p;
                p.x = data[2 * indices[iter * dataSize + d]];
                p.y = data[2 * indices[iter * dataSize + d] + 1];
                // double dist = distance(m, data[indices[iter * dataSize + d]]);
                double dist = distance(m, p);
                sum += dist * dist;
            }
            double error = sum / trainSize;

            if (error < minError)
            {
                minError = error;
                bestModel = m;
            }
        }
        bestCandidateModels[iter] = bestModel;
    }

    free(candidateModels);

    // evaluate the models
    int numGood[maxIter];
    int tmpInlierIndices[dataSize - trainSize];
    for (int iter = 0; iter < maxIter; iter++)
    {
        numGood[iter] = checkGood(bestCandidateModels, data, dataSize, indices, trainSize, thresh, iter, tmpInlierIndices);
        if (numGood[iter] < wellCount)
        {
            numGood[iter] = 0;
        }
        else
        {
            // reorder (put the new inliers first)
            for (int g = 0; g < numGood[iter]; g++)
            {
                int tmp = indices[iter * dataSize + trainSize + g];
                indices[iter * dataSize + trainSize + g] = indices[iter * dataSize + tmpInlierIndices[g]];
                indices[iter * dataSize + tmpInlierIndices[g]] = tmp;
            }
        }
    }

    // fit again using the new inlier indices
    int numInlierComb = triangMax(dataSize);
    struct LinearModel *inlierModels = (struct LinearModel*)malloc(maxIter * numInlierComb * sizeof(struct LinearModel));
    struct LinearModel *d_inlierModels;
    cudaMalloc(&d_inlierModels, maxIter * numInlierComb * sizeof(inlierModels[0]));

    for (int iter = 0; iter < maxIter; iter++)
    {
        if (numGood[iter] == 0)
            continue;

        int numComb = triangMax(trainSize + numGood[iter]);
        for (int t = 0; t < numComb; t++)
        {
            buildModel<<<1, numComb>>>(d_data, dataSize, d_indices, iter, numComb, d_inlierModels);
        }
    }

    cudaMemcpy(inlierModels, d_inlierModels, maxIter * numInlierComb * sizeof(inlierModels[0]), cudaMemcpyDeviceToHost);
    cudaFree(d_inlierModels);

    // find the best model for each iteration
    struct LinearModel bestInlierModel[maxIter];
    double minInlierError[maxIter];
    for (int iter = 0; iter < maxIter; iter++)
    {
        if (numGood[iter] == 0)
        {
            minInlierError[iter] = std::numeric_limits<double>::infinity();
            continue;
        }
        double minError = std::numeric_limits<double>::infinity();
        struct LinearModel bestModel;
        // reduce
        int numTrainAndGood = trainSize + numGood[iter];
        int numComb = triangMax(numTrainAndGood);
        for (int t = 0; t < numComb; t++)
        {
            int i, k;
            tuple_triang(t, i, k);
            struct LinearModel m = inlierModels[iter * numInlierComb + t];

            double sum = 0;
            // reduce
            for (int d = 0; d < numTrainAndGood; d++)
            {
                struct Point p;
                p.x = data[2 * indices[iter * dataSize + d]];
                p.y = data[2 * indices[iter * dataSize + d] + 1];
                // double dist = distance(m, data[indices[iter * dataSize + d]]);
                double dist = distance(m, p);
                sum += dist * dist;
            }
            double error = sum / numTrainAndGood;

            if (error < minError)
            {
                minError = error;
                bestModel = m;
            }
        }
        bestInlierModel[iter] = bestModel;
        minInlierError[iter] = minError;
    }

    free(inlierModels);

    // find the best model
    struct LinearModel finalModel;
    double finalError = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < maxIter; iter++)
    {
        if (minInlierError[iter] < finalError)
        {
            finalError = minInlierError[iter];
            finalModel = bestInlierModel[iter];
        }
    }

    cudaFree(d_data);
    cudaFree(d_indices);

    std::cout << "error: " << finalError << std::endl;

    if (finalError == std::numeric_limits<double>::infinity())
    {
        std::cerr << "RANSAC failed" << std::endl;
    }
    return finalModel;
}

std::vector<double> readCSV(const char *path)
{
    std::ifstream file(path);
    std::vector<double> data;

    std::string line;
    while (getline(file, line))
    {
        double x, y;
        sscanf(line.c_str(), "%lf,%lf", &x, &y);
        data.push_back(x);
        data.push_back(y);
    }
    file.close();

    return data;
}

int main(int argc, char const *argv[])
{
    // srand(time(NULL));
    srand(420);
    std::vector<double> data = readCSV("points.csv");
    int dataSize = data.size() / 2;

    clock_t t0 = clock();
    const int numIters = 4;
    float trainRatio = .3f;
    float wellRatio = .1f;
    double errorThresh = .3;
    struct LinearModel m = ransac(&data[0], dataSize, numIters, errorThresh, (int)(trainRatio * dataSize), (int)(wellRatio * dataSize));
    clock_t t1 = clock();
    double elapsed_secs = double(t1 - t0) / CLOCKS_PER_SEC;

    std::cout << "Best Model (slope, intercept): " << m.slope << " " << m.intercept << std::endl;
    std::cout << "Time taken: " << elapsed_secs << "s" << std::endl;
    return 0;
}
