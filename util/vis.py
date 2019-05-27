#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys

def abline(slope, intercept):
    x = np.array(plt.gca().get_xlim())
    y = slope * x + intercept
    plt.plot(x, y, '--')

def show(csv_path: str, slope: float, intercept: float):
    data = np.loadtxt(csv_path, delimiter=",")

    if data.shape[1] == 3:
        categories = data[:, 2].astype(np.uint8)
        colour_map = cm.rainbow(np.linspace(0, 1, categories.max() + 1))
        colours = [colour_map[c] for c in categories]
    else:
        colours = None

    plt.scatter(data[:, 0], data[:, 1], color=colours)
    abline(slope, intercept)
    plt.show()

    plt.savefig("vis.jpg")

if __name__ == "__main__":
    show("points.csv", float(sys.argv[1]), float(sys.argv[2]))
