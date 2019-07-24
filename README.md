An implementation of the RANSAC algorithm on the GPU using NVIDIA CUDA.

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
# creates a fit.jpg image that shows the result
python3 src/fit.py

# visualisation of the RANSAC step (no fit)
# creates a vis.jpg image
python3 util/vis.py
```

The programs accept only hard-coded file paths, therefore the generated files must not be renamed or moved.
