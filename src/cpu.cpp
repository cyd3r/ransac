#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

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

double mse(LinearModel model, std::vector<Point> data)
{
    double sum = 0;
    for (auto p : data)
    {
        double d = distance(model, p);
        sum += d * d;
    }
    return sum / data.size();
}

LinearModel fitModel(std::vector<Point> data)
{
    LinearModel best;
    double minError = std::numeric_limits<double>::infinity();
    // try every combination of two points to find a good line
    for (int i = 0; i < data.size(); i++)
    {
        for (int k = 0; k < data.size(); k++)
        {
            if (i == k)
                continue;

            Point pI, pK;
            pI = data[i];
            pK = data[k];

            LinearModel m;
            m.slope = (pK.y - pI.y) / (pK.x - pI.x);
            m.intercept = pI.y - m.slope * pI.x;
            double error = mse(m, data);
            if (error < minError)
            {
                minError = error;
                best = m;
            }
        }
    }
    return best;
}

LinearModel ransac(std::vector<Point> data, int maxIter, double thresh, int trainSize, int wellCount)
{
    double minError = std::numeric_limits<double>::infinity();
    LinearModel bestModel;
    for (int iter = 0; iter < maxIter; iter++)
    {
        // draw `trainSize` samples and keep the rest. This is equivalent to shuffling and taking the first items
        std::random_shuffle(data.begin(), data.end());

        std::vector<Point> good;
        std::vector<Point> trainSet(data.begin(), data.begin() + trainSize);

        LinearModel model = fitModel(trainSet);

        for (int i = trainSize; i < data.size(); i++)
        {
            Point p = data[i];
            if (distance(model, p) < thresh)
            {
                good.push_back(p);
            }
        }

        if (good.size() > wellCount)
        {
            // merge original train set and the good set
            trainSet.insert(trainSet.end(), good.begin(), good.end());
            model = fitModel(trainSet);
            double error = mse(model, trainSet);
            if (error < minError)
            {
                minError = error;
                bestModel = model;
            }
        }
    }
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
    std::vector<Point> data = readCSV("points.csv");

    clock_t t0 = clock();
    LinearModel m = ransac(data, 50, 0.3, 30, 10);
    clock_t t1 = clock();
    double elapsed_secs = double(t1 - t0) / CLOCKS_PER_SEC;

    std::cout << "Best Model (slope, intercept): " << m.slope << ", " << m.intercept << std::endl;
    std::cout << "Time taken: " << elapsed_secs << std::endl;
    return 0;
}
