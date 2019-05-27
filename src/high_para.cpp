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

double distance(LinearModel model, Point p)
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

int * buildIndices(std::vector<Point> data, int maxIter)
{
    int *indices = (int*)malloc(maxIter * data.size() * sizeof(int));
    for (int iter = 0; iter < maxIter; iter++)
    {
        for (int idx = 0; idx < data.size(); idx++)
        {
            indices[iter * data.size() + idx] = idx;
        }
        shuffle(&indices[iter * data.size()], data.size());
    }
    return indices;
}

LinearModel buildModel(std::vector<Point> data, int dataSize, int *indices, int iter, int i, int k)
{
    Point pI, pK;
    pI = data[indices[iter * dataSize + i]];
    pK = data[indices[iter * dataSize + k]];

    LinearModel m;
    m.slope = (pK.y - pI.y) / (pK.x - pI.x);
    m.intercept = pI.y - m.slope * pI.x;
    return m;
}

int checkGood(LinearModel *bestIter, std::vector<Point> data, int dataSize, int *indices, int trainSize, double thresh, int iter, int *good)
{
    LinearModel m = bestIter[iter];
    // reduce
    // int good[dataSize - trainSize];
    int goodSize = 0;
    for (int d = trainSize; d < dataSize; d++)
    {
        double dist = distance(m, data[indices[iter * dataSize + d]]);
        if (dist < thresh)
        {
            good[goodSize] = d;
            goodSize++;
        }
    }

    return goodSize;
}

LinearModel ransac(std::vector<Point> data, int maxIter, double thresh, int trainSize, int wellCount)
{
    int *indices = buildIndices(data, maxIter);
    int dataSize = data.size();

    // the number of unique combinations of all data points
    int numCombinations = triangMax(trainSize);

    // produce every possible model
    LinearModel candidateModels[maxIter * numCombinations];
    for (int iter = 0; iter < maxIter; iter++)
    {
        for (int t = 0; t < numCombinations; t++)
        {
            int i, k;
            tuple_triang(t, i, k);
            candidateModels[iter * numCombinations + t] = buildModel(data, dataSize, indices, iter, i, k);
        }
    }

    // find the best model for each iteration
    LinearModel bestCandidateModels[maxIter];
    for (int iter = 0; iter < maxIter; iter++)
    {
        double minError = std::numeric_limits<double>::infinity();
        LinearModel bestModel;
        // reduce
        for (int t = 0; t < numCombinations; t++)
        {
            int i, k;
            tuple_triang(t, i, k);
            LinearModel m = candidateModels[iter * numCombinations + t];
            double sum = 0;
            // reduce
            for (int d = 0; d < trainSize; d++)
            {
                double dist = distance(m, data[indices[iter * dataSize + d]]);
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
            // reorder
            for (int g = 0; g < numGood[iter]; g++)
            {
                int tmp = indices[iter * dataSize + trainSize + g];
                indices[iter * dataSize + trainSize + g] = indices[iter * dataSize + tmpInlierIndices[g]];
                indices[iter * dataSize + tmpInlierIndices[g]] = tmp;
            }
        }
    }

    // fit again using the inlier indices
    int numInlierComb = triangMax(dataSize);
    LinearModel inlierModels[maxIter * numInlierComb];
    for (int iter = 0; iter < maxIter; iter++)
    {
        if (numGood[iter] == 0)
            continue;

        int numComb = triangMax(trainSize + numGood[iter]);
        for (int t = 0; t < numComb; t++)
        {
            int i, k;
            tuple_triang(t, i, k);
            inlierModels[iter * numInlierComb + t] = buildModel(data, dataSize, indices, iter, i, k);
        }
    }

    // find the best model for each iteration
    LinearModel bestInlierModel[maxIter];
    double minInlierError[maxIter];
    for (int iter = 0; iter < maxIter; iter++)
    {
        if (numGood[iter] = 0)
        {
            minInlierError[iter] = std::numeric_limits<double>::infinity();
            continue;
        }
        double minError = std::numeric_limits<double>::infinity();
        LinearModel bestModel;
        // reduce
        int numTrainAndGood = trainSize + numGood[iter];
        int numComb = triangMax(numTrainAndGood);
        for (int t = 0; t < numComb; t++)
        {
            int i, k;
            tuple_triang(t, i, k);
            LinearModel m = inlierModels[iter * numInlierComb + t];

            double sum = 0;
            // reduce
            for (int d = 0; d < numTrainAndGood; d++)
            {
                double dist = distance(m, data[indices[iter * dataSize + d]]);
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

    // find the best model
    LinearModel finalModel;
    double finalError = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < maxIter; iter++)
    {
        if (minInlierError[iter] < finalError)
        {
            finalError == minInlierError[iter];
            finalModel = bestInlierModel[iter];
        }
    }

    std::cout << finalError << std::endl;
    return finalModel;
}

std::vector<Point> readCSV(const char *path)
{
    std::ifstream file(path);
    std::vector<Point> data;

    std::string line;
    while (getline(file, line))
    {
        Point p;
        sscanf(line.c_str(), "%lf,%lf", &p.x, &p.y);
        data.push_back(p);
    }
    file.close();

    return data;
}

int main(int argc, char const *argv[])
{
    // srand ( time(NULL) );
    std::vector<Point> data = readCSV("points.csv");

    clock_t t0 = clock();
    LinearModel m = ransac(data, 50, 0.3, 30, 10);
    clock_t t1 = clock();
    double elapsed_secs = double(t1 - t0) / CLOCKS_PER_SEC;

    std::cout << "Best Model (slope, intercept): " << m.slope << ", " << m.intercept << std::endl;
    std::cout << "Time taken: " << elapsed_secs << "s" << std::endl;
    return 0;
}
