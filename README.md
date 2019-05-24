An implementation of the RANSAC algorithm. The goal is to provide a GPU implementation as well and compare the two.

## Requirements

- numpy
- matplotlib

## Steps

``` sh
# generate random data points
python3 util/generate.py

# compile the RANSAC implementation
g++ src/main.cpp
# and run it!
./a.out
```
