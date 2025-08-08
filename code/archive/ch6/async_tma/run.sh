#!/usr/bin/env bash
nvcc -std=c++17 -DCUDA_VERSION=12.9 -DCUDA_VERSION=12.9 -DCUDA_VERSION=13.1 -DCUDA_VERSION=13.1 -arch=sm_100a -O3 kernelWithTMA.cu -o kernel_tma && ./kernel_tma
