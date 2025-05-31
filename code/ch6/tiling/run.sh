#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 naiveMatMul.cu -o naive && ./naive
nvcc -std=c++17 -arch=sm_90 -O3 tiledMatMul.cu -o tiled && ./tiled
