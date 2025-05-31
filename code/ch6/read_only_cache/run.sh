#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 naiveLookup.cu -o naive_lookup && ./naive_lookup
nvcc -std=c++17 -arch=sm_90 -O3 ldgLookup.cu -o ldg_lookup   && ./ldg_lookup
