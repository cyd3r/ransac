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

# only use a subset for plotting (too many data points will clutter the plot with useless information)
max_datapoints = 1000
if len(data) > max_datapoints:
    plot_data = data[::len(data) // max_datapoints]
    is_inlier = np.array([i in inliers for i in range(0, len(data), len(data) // max_datapoints)])
else:
    plot_data = data
    is_inlier = np.array([i in inliers for i in range(len(plot_data))])

categories = plot_data[:, 2].astype(np.uint8)

for c in np.unique(categories):
    sel = (categories == c) & (~is_inlier)
    plt.scatter(plot_data[sel, 0], plot_data[sel, 1], label="noise" if c == 0 else "linear")
plt.scatter(plot_data[is_inlier, 0], plot_data[is_inlier, 1], label="inliers")

x = np.arange(20 + 1)
y = model.predict(x.reshape(-1, 1))
plt.plot(x, y)

plt.legend()
plt.show()

plt.savefig("fit.jpg")
