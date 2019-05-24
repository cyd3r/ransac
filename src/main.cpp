#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
// required?
#include <stdio.h>

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

double distance(LinearModel model, Point p) {
    return std::abs(model.slope * p.x - p.y + model.intercept) / std::sqrt(model.slope * model.slope + 1);
}

double mse(LinearModel model, std::vector<Point> data) {
    double sum = 0;
    for (auto p : data) {
        double d = distance(model, p);
        sum += d * d;
    }
    return sum / data.size();
}

LinearModel fitModel(std::vector<Point> data) {
    LinearModel best;
    double bestError = std::numeric_limits<double>::infinity();
    // try every combination of two points to find a good line
    for (int i = 0; i < data.size(); i++) {
        for (int k = 0; k < data.size(); k++) {
            if (i == k)
                continue;

            Point pI, pK;
            pI = data[i];
            pK = data[k];

            LinearModel m;
            m.slope = (pK.y - pI.y) / (pK.x - pI.x);
            m.intercept = pI.y - m.slope * pI.x;
            double error = mse(m, data);
            if (error < bestError) {
                bestError = error;
                best = m;
            }
        }
    }
    return best;
}

void printModel(LinearModel model) {
    std::cout << "slope: " << model.slope << ", intercept: " << model.intercept << std::endl;
}

LinearModel ransac(std::vector<Point> data, int maxIter, double thresh, int trainSize, int wellCount) {
    auto random = std::mt19937{std::random_device{}()};
    // TODO: something really large
    double minError = std::numeric_limits<double>::infinity();
    LinearModel bestModel;
    for (int iter = 0; iter < maxIter; iter++) {
        // draw `trainSize` samples and keep the rest. This is equivalent to shuffling and taking the first items
        std::random_shuffle(data.begin(), data.end());

        std::vector<Point> good;
        std::vector<Point> trainSet(data.begin(), data.begin() + trainSize);

        LinearModel model = fitModel(trainSet);

        for (int i = trainSize; i < data.size(); i++) {
            Point p = data[i];
            if (distance(model, p) < thresh) {
                good.push_back(p);
            }
        }

        std::cout << "good " << good.size();

        if (good.size() > wellCount) {
            trainSet.insert(trainSet.end(), good.begin(), good.end());
            model = fitModel(trainSet);
            double error = mse(model, trainSet);
            if (error < minError) {
                minError = error;
                bestModel = model;
            }
        }
        std::cout << " error: " << minError << " ";
        printModel(bestModel);
    }
    return bestModel;
}

std::vector<Point> readCSV(const char *path)
{
    std::ifstream file(path);

    std::vector<Point> data;

    std::string line = "";
    // Iterate through each line and split the content using delimeter
    while (getline(file, line))
    {
        Point p;
        sscanf(line.c_str(), "%lf,%lf", &p.x, &p.y);

        data.push_back(p);
    }
    // Close the File
    file.close();

    return data;
}

int main(int argc, char const *argv[])
{
    // LinearModel m;
    // m.slope = 2;
    // m.intercept = .3;
    // Point p;
    // p.x = 2.3;
    // p.y = 5.6;
    // std::cout << "eval: " << distance(m, p) << std::endl;

    // return 0;

    std::vector<Point> data = readCSV("points.csv");
    // for (auto p : data) {
    //     std::cout << "x: " << p.x << ", y: " << p.y << std::endl;
    // }

    // return 0;

    LinearModel m = ransac(data, 100, 0.3, 30, 10);
    return 0;
}
