#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

int triangMax(int n)
{
    return (n * (n - 1)) / 2;
}

struct dist_functor {
    double slope = 0;
    double intercept = 0;
    dist_functor(const double3 model)
    {
        slope = model.x;
        intercept = model.y;
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

double3 getInfModel()
{
    double3 m;
    m.z = std::numeric_limits<double>::infinity();
    return m;
}

__device__ void triangToTuple(int t, int &x, int &y)
{
    y = (int)((1 + (int)sqrtf(1 + 8 * t)) / 2);
    x = t - (y * (y - 1)) / 2;
}

__global__ void buildModel(double2 *data, int numCombinations, double3 *models)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int i, k;
    triangToTuple(t, i, k);

    double2 pI = data[i];
    double2 pK = data[k];

    double3 m;
    m.x = (pK.y - pI.y) / (pK.x - pI.x);
    m.y = pI.y - m.x * pI.x;

    models[t] = m;
}

double3 findBest(int trainSize, double3 *candidates, int candidatesSize, double2 *data, double2 *d_data)
{
    // reduce
    for (int t = 0; t < candidatesSize; t++)
    {
        if (t % 1000 == 0)
            std::cout << t + 1 << "/" << candidatesSize << std::endl;
        double3 m = candidates[t];
        struct dist_functor dist_op = dist_functor(m);
        double sum = thrust::transform_reduce(data, data + trainSize, dist_op, 0.0, thrust::plus<double>());
        // double sum = thrust::transform_reduce(thrust::device, d_data, d_data + trainSize, dist_op, 0.0, thrust::plus<double>());
        // mean squarred error
        candidates[t].z = sum / trainSize;
    }

    std::cout << "end reduce" << std::endl;

    struct min_model min_op;
    double3 bestM = thrust::reduce(candidates, candidates + candidatesSize, getInfModel(), min_op);
    return bestM;
}

void fit(double2 *rawData, int *inliers, int inliersSize, double &slope, double &intercept)
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

    double3 *inlierModels = (double3*)malloc(numInlierComb * sizeof(double3));
    double3 *d_inlierModels;
    cudaMalloc(&d_inlierModels, numInlierComb * sizeof(inlierModels[0]));

    std::cout << "build" << std::endl;

    buildModel<<<numBlocks, threadsPerBlock>>>(d_data, numInlierComb, d_inlierModels);

    cudaMemcpy(inlierModels, d_inlierModels, numInlierComb * sizeof(inlierModels[0]), cudaMemcpyDeviceToHost);
    cudaFree(d_inlierModels);

    std::cout << "reduce" << std::endl;

    double3 model = findBest(inliersSize, inlierModels, numInlierComb, data, d_data);

    free(inlierModels);
    cudaFree(d_data);

    slope = model.x;
    intercept = model.y;
}
