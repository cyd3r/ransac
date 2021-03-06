#!/usr/bin/env python3

import sys
import numpy as np

def random_linear2d(slope: float, intercept: float, error: float, width: float):
    noise = np.random.uniform(-error, error)

    x = np.random.uniform(width)
    y = slope * x + intercept + noise
    return np.array([x, y])

def generate(out_path: str, samples: int, linear_ratio: float, slope: float, intercept: float, linear_error: float, dimensions: np.array):
    assert linear_ratio >= 0 and linear_ratio <= 1
    # generate noise
    num_linear = int(samples * linear_ratio)
    num_noise = samples - num_linear

    # noise = uniform distributed
    noise = [(*np.random.uniform(high=dimensions), 0) for _ in range(num_noise)]

    linear = [(*random_linear2d(slope, intercept, linear_error, dimensions[0]), 1) for _ in range(num_linear)]

    data = np.vstack(noise + linear)
    np.random.shuffle(data)

    np.savetxt(out_path, data, delimiter=",")

if __name__ == "__main__":
    num_samples = int(sys.argv[1])
    generate("points.csv", num_samples, .7, .8, 0, .3, np.array([20, 20]))
