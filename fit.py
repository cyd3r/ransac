#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.linear_model import LinearRegression

def abline(slope, intercept):
    x = np.array(plt.gca().get_xlim())
    y = slope * x + intercept
    plt.plot(x, y, '--')

data = np.loadtxt("points.csv", delimiter=",")
inliers = np.loadtxt("inliers.txt", np.int)

fit_data = data[inliers]

model = LinearRegression()
model.fit(fit_data[:, 0].reshape(-1, 1), fit_data[:, 1].reshape(-1, 1))
# model = np.polyfit(fit_data[0,:], fit_data[1,:], 1)

if data.shape[1] == 3:
    categories = data[:, 2].astype(np.uint8)
    colour_map = cm.rainbow(np.linspace(0, 1, categories.max() + 2))
    colours = [colour_map[c] for c in categories]

    colours = [colour_map[categories.max() + 1] if i in inliers else c for i, c in enumerate(colours)]
else:
    colours = None

plt.scatter(data[:, 0], data[:, 1], color=colours)
x = np.arange(20)
y = model.predict(x.reshape(-1, 1))
plt.plot(x, y)

# abline(slope, intercept)
plt.title("Slope: {slope:.4}, Intercept: {intercept:.4}, Error: {error:.4}")
plt.show()

plt.savefig("fit.jpg")