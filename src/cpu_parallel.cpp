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

void computeDistances(LinearModel model, std::vector<Point> data, int *index, int indexSize, double *distances)
{
    for (int i = 0; i < indexSize; i++)
    {
        distances[i] = distance(model, data[index[i]]);
    }
}

double mse(LinearModel model, std::vector<Point> data, int *index, int indexSize)
{
    double sum = 0;
    for (int i = 0; i < indexSize; i++)
    {
        double d = distance(model, data[index[i]]);
        sum += d * d;
    }
    return sum / indexSize;
}

double mse(double *distances, int size)
{
    // mean squared error
    double sum = 0;
    for (int i = 0; i < size; i++)
    {
        sum += distances[i] * distances[i];
    }
    return sum / size;
}

LinearModel fitModel(std::vector<Point> data, int *index, int indexSize)
{
    LinearModel best;
    double minError = std::numeric_limits<double>::infinity();
    // try every combination of two points to find a good line
    int numCombinations = (indexSize * (indexSize - 1)) / 2;
    for (int t = 0; t < numCombinations; t++)
    {
        int i, k;
        tuple_triang(t, i, k);

        Point pI, pK;
        pI = data[index[i]];
        pK = data[index[k]];

        LinearModel m;
        m.slope = (pK.y - pI.y) / (pK.x - pI.x);
        m.intercept = pI.y - m.slope * pI.x;

        double distances[indexSize];
        computeDistances(m, data, index, indexSize, distances);
        double error = mse(distances, indexSize);
        // double error = mse(m, data, index, indexSize);

        if (error < minError)
        {
            minError = error;
            best = m;
        }
    }
    return best;
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

LinearModel ransac(std::vector<Point> data, int maxIter, double thresh, int trainSize, int wellCount)
{
    int evalSize = data.size() - trainSize;
    // first, generate shuffled indices
    int *indices = buildIndices(data, maxIter);

    double minError = std::numeric_limits<double>::infinity();
    LinearModel bestModel;
    for (int iter = 0; iter < maxIter; iter++)
    {
        // the data used to fit the model
        int *idx = &indices[iter * data.size()];
        // the data used to evaluate the model (complement of idx)
        int *evalIdx = &indices[iter * data.size() + trainSize];

        LinearModel model = fitModel(data, idx, trainSize);

        // find inliers for the current model
        double distances[evalSize];
        computeDistances(model, data, evalIdx, evalSize, distances);
        std::vector<int> good;
        for (int i = 0; i < evalSize; i++)
        {
            if (distances[i] < thresh)
            {
                good.push_back(evalIdx[i]);
            }
        }

        int goodCount = good.size();

        if (goodCount > wellCount)
        {
            std::vector<int> idx2(idx, idx + trainSize);
            idx2.insert(idx2.end(), good.begin(), good.end());
            // merge original train set and the good set
            model = fitModel(data, &idx2[0], idx2.size());
            computeDistances(model, data, &idx2[0], idx2.size(), distances);
            double error = mse(distances, idx2.size());
            // double error = mse(model, data, &idx2[0], idx2.size());
            if (error < minError)
            {
                minError = error;
                bestModel = model;
            }
        }
    }
    free(indices);
    return bestModel;
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
    // srand(time(NULL));
    srand(420);
    std::vector<Point> data = readCSV("points.csv");

    clock_t t0 = clock();
    LinearModel m = ransac(data, 50, 0.3, 30, 10);
    clock_t t1 = clock();
    double elapsed_secs = double(t1 - t0) / CLOCKS_PER_SEC;

    std::cout << "Best Model (slope, intercept): " << m.slope << ", " << m.intercept << std::endl;
    std::cout << "Time taken: " << elapsed_secs << "s" << std::endl;
    return 0;
}
