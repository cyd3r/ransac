#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def abline(slope, intercept):
    x = np.array(plt.gca().get_xlim())
    y = slope * x + intercept
    plt.plot(x, y, '--')

def show(csv_path: str):
    data = np.loadtxt(csv_path, delimiter=",")

    if data.shape[1] == 3:
        categories = data[:, 2].astype(np.uint8)
        colour_map = cm.rainbow(np.linspace(0, 1, categories.max() + 1))
        colours = [colour_map[c] for c in categories]
    else:
        colours = None

    plt.scatter(data[:, 0], data[:, 1], color=colours)
    abline(0.860695, -0.248514)
    plt.show()

if __name__ == "__main__":
    show("points.csv")
