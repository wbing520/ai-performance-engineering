#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 transposeNaive.cu -o naive  && ./naive
nvcc -std=c++17 -arch=sm_90 -O3 transposePadded.cu -o padded && ./padded
