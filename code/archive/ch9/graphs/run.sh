#!/usr/bin/env bash
set -euo pipefail

nvcc -std=c++17 -O3 -arch=sm_100 -DCUDA_VERSION=12.8 -lnvtx3 graph_capture.cu -o graph_capture
./graph_capture
python3 pytorch_graph.py
