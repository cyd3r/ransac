#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def abline(slope, intercept):
    x = np.array(plt.gca().get_xlim())
    y = slope * x + intercept
    plt.plot(x, y, '--')

def plot_points(data, inliers):
    categories = data[:, 2].astype(np.uint8)
    unique_cats = np.unique(categories)
    for c in unique_cats:
        data[categories == c]
        plt.scatter(data[categories == c, 0], data[categories == c, 1], label="noise" if c == 0 else "linear")

def show(csv_path: str, slope: float, intercept: float, error: float):
    data = np.loadtxt(csv_path, delimiter=",")
    inliers = np.loadtxt("inliers.txt", np.int)

    # only use a subset for plotting
    max_datapoints = 1000
    if len(data) > max_datapoints:
        data = data[::(len(data) // max_datapoints)]

    plot_points(data, inliers)
    abline(slope, intercept)
    plt.title(f"Slope: {slope:.4}, Intercept: {intercept:.4}, Error: {error:.4}")
    plt.legend()
    plt.show()

    plt.savefig("vis.jpg")

if __name__ == "__main__":
    with open("results.txt", "r") as f:
        model = [float(s) for s in f.readline().split(" ")]
    show("points.csv", *model)
