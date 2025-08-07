#!/usr/bin/env bash
nvcc -std=c++17 -DCUDA_VERSION=12.9 -DCUDA_VERSION=12.9 -DCUDA_VERSION=13.1 -DCUDA_VERSION=13.1 -arch=sm_100a -O3 graph_capture.cu -o graph_capture
./graph_capture
python3 pytorch_graph.py
