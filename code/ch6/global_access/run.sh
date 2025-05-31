#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 uncoalescedCopy.cu -o uncoalesced && ./uncoalesced
nvcc -std=c++17 -arch=sm_90 -O3 coalescedCopy.cu -o coalesced   && ./coalesced
