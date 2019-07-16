An implementation of the RANSAC algorithm. The goal is to provide a GPU implementation as well and compare the two.

## Requirements

- python3
    - numpy
    - matplotlib
    - sklearn

## Steps

``` sh
# generate random data points
python3 util/generate.py 1000000

# compile the RANSAC implementation
make

# and run it!
# with CUDA
./gpu.out
# without CUDA
./host.out

# run a fit on the extracted inliers
python3 src/fit.py

# visualisation
python3 util/vis.py
```
