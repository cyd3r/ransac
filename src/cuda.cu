#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

// #include "fit.cu"

struct LinearModel
{
    double slope;
    double intercept;
    double error;
};

struct LinearModel maxModel(struct LinearModel a, struct LinearModel b)
{
    return a.error > b.error ? a : b;
}

double distance(struct LinearModel model, double2 p)
{
    return std::abs(model.slope * p.x - p.y + model.intercept) / std::sqrt(model.slope * model.slope + 1);
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
    std::cout << inliers[0] << " " << inliers[1] << " " << numGood << std::endl;
    *inliersSize = numGood + 2;
    if (numGood < wellCount)
    {
        std::cout << "num good: " << numGood << "<" << wellCount << std::endl;
        struct LinearModel fail;
        fail.slope = std::numeric_limits<double>::quiet_NaN();
        fail.intercept = std::numeric_limits<double>::quiet_NaN();
        fail.error = -1;
        return fail;
    }

    bestModel.error = numGood;

    cudaFree(d_data);

    return bestModel;
}

struct LinearModel ransac(double2 *data, int dataSize, int maxIter, double thresh, int wellCount)
{
    struct LinearModel best;
    int *inliers = (int *)malloc(dataSize * sizeof(int));
    int *bestInliers = (int *)malloc(dataSize * sizeof(int));
    int bestInliersSize = 0;
    int inliersSize;
    best.error = -1;
    int bestIter = -1;
    for (int iter = 0; iter < maxIter; iter++)
    {
        struct LinearModel m = singleIter(iter, data, dataSize, thresh, wellCount, inliers, &inliersSize);
        std::cout << iter << " " << m.slope << " " << m.intercept << " " << m.error << std::endl;
        if (m.error > best.error)
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
    fprintf(f, "%f %f %f\n", m.slope, m.intercept, m.error);
    fclose(f);
    return 0;
}
