#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1

# Latest Nsight Systems profiling for CUDA 12.9, PyTorch 2.8, and Blackwell B200/B300
nsys profile \
  --force-overwrite=true \
  -o $(basename $SCRIPT)_nsys \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --cudabacktrace=true \
  --cudabacktrace-threshold=0 \
  --gpu-metrics-device=all \
  --stats=true \
  python3 $SCRIPT
