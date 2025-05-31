#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 copyScalar.cu -o scalar  && ./scalar
nvcc -std=c++17 -arch=sm_90 -O3 copyVector.cu -o vector  && ./vector
