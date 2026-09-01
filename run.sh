#!/bin/bash
set -e
cd "$(dirname "$0")"
nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -fopenmp -o main main.cu
./main
