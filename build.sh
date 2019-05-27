#!/bin/sh
echo cpu
g++ -o cpu.out src/cpu.cpp
echo cpu_parallel
g++ -o para.out src/cpu_parallel.cpp
echo high_para
g++ -o high.out src/high_para.cpp
