#!/usr/bin/env bash
nvcc -std=c++17 -DCUDA_VERSION=12.9 -lnvtx3 -DCUDA_VERSION=12.9 -DCUDA_VERSION=12.9 -DCUDA_VERSION=13.1 -DCUDA_VERSION=13.1 -arch=sm_100a -O3 naiveLookup.cu -o naive_lookup && ./naive_lookup
nvcc -std=c++17 -DCUDA_VERSION=12.9 -lnvtx3 -DCUDA_VERSION=12.9 -DCUDA_VERSION=12.9 -DCUDA_VERSION=13.1 -DCUDA_VERSION=13.1 -arch=sm_100a -O3 ldgLookup.cu -o ldg_lookup   && ./ldg_lookup
