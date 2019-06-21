import os
import time
import numpy as np
import matplotlib.pyplot as plt
from generate import generate

if __name__ == "__main__":
    nums = np.array([1e+3, 1e+4, 3e+4, 1e+5, 5e+5, 1e+7], dtype=np.int)
    gpu_times = []
    host_times = []
    for num_points in nums:
        print("Generate", num_points, "data points")
        generate("points.csv", num_points, .7, .8, 0, .3, np.array((20, 20)))

        print("GPU")
        t0 = time.time()
        os.system("./gpu.out")
        t1 = time.time()
        gpu_times.append(t1 - t0)

        print("Host")
        t0 = time.time()
        os.system("./host.out")
        t1 = time.time()
        host_times.append(t1 - t0)

    host_times = np.array(host_times)
    gpu_times = np.array(gpu_times)

    plt.plot(nums, gpu_times, label="GPU")
    plt.plot(nums, host_times, label="Host")
    plt.legend()
    plt.xlabel("Number of points")
    plt.ylabel("Time taken")
    plt.savefig("times.jpg")
    plt.close()

    plt.plot(nums, host_times / gpu_times)
    plt.xlabel("Number of points")
    plt.xscale("log")
    plt.ylabel("Speeup")
    plt.savefig("speedup.jpg")
    plt.close()

    np.savetxt("times.csv", np.stack([nums, gpu_times, host_times]), delimiter=",")
