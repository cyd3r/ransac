#!/bin/sh
echo gpu
nvcc -o gpu.out -DUSE_GPU=1 src/cuda.cu
echo host
nvcc -o host.out -DUSE_GPU=0 src/cuda.cu
