#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def abline(slope, intercept):
    x = np.array(plt.gca().get_xlim())
    y = slope * x + intercept
    plt.plot(x, y, '--')

def show(csv_path: str, slope: float, intercept: float, error: float):
    data = np.loadtxt(csv_path, delimiter=",")

    inliers = np.loadtxt("inliers.txt", np.int)

    if data.shape[1] == 3:
        categories = data[:, 2].astype(np.uint8)
        colour_map = cm.rainbow(np.linspace(0, 1, categories.max() + 2))
        colours = [colour_map[c] for c in categories]

        colours = [colour_map[categories.max() + 1] if i in inliers else c for i, c in enumerate(colours)]
    else:
        colours = None

    plt.scatter(data[:, 0], data[:, 1], color=colours)
    abline(slope, intercept)
    plt.title(f"Slope: {slope:.4}, Intercept: {intercept:.4}, Error: {error:.4}")
    plt.show()

    plt.savefig("vis.jpg")

if __name__ == "__main__":
    with open("results.txt", "r") as f:
        model = [float(s) for s in f.readline().split(" ")]
    show("points.csv", *model)
