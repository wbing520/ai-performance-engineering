#!/usr/bin/env bash
nvcc -std=c++17 -DCUDA_VERSION=12.9 -lnvtx3 -DCUDA_VERSION=12.9 -lnvtx3 -DCUDA_VERSION=12.9 -DCUDA_VERSION=12.9 -DCUDA_VERSION=13.1 -DCUDA_VERSION=13.1 -arch=sm_100a -O3 uncoalescedCopy.cu -o uncoalesced && ./uncoalesced
nvcc -std=c++17 -DCUDA_VERSION=12.9 -lnvtx3 -DCUDA_VERSION=12.9 -lnvtx3 -DCUDA_VERSION=12.9 -DCUDA_VERSION=12.9 -DCUDA_VERSION=13.1 -DCUDA_VERSION=13.1 -arch=sm_100a -O3 coalescedCopy.cu -o coalesced   && ./coalesced
