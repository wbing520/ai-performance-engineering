#!/usr/bin/env bash
nvcc -std=c++17 -arch=sm_90 -O3 graph_capture.cu -o graph_capture
./graph_capture
python3 pytorch_graph.py
