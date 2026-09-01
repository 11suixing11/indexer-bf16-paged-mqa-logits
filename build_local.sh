#!/usr/bin/env bash
# Local dev build. run.sh is left untouched (it hardcodes -arch=sm_86 for the RTX 3050
# the task was authored on); this builds a fat binary that runs on both sm_86 and the
# sm_120 (Blackwell) dev GPU, and keeps -lineinfo so ncu can attribute per-line stalls.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-13.3/bin:$PATH

nvcc -O3 -std=c++17 -lineinfo \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_120,code=sm_120 \
     -Xcompiler -fopenmp \
     -o main main.cu

echo "build ok -> ./main"
