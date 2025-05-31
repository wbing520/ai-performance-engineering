#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 kernelWithTMA.cu -o kernel_tma && ./kernel_tma
