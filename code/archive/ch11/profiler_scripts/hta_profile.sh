#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1

# Holistic Tracing Analysis (HTA) for multi-GPU profiling
echo "Running HTA profiler for multi-GPU analysis..."

# HTA configuration for CUDA 12.8 and Blackwell B200/B300
nsys profile \
  --force-overwrite=true \
  -o $(basename $SCRIPT)_hta \
  -t cuda,nvtx,osrt,cudnn,cublas,nccl \
  -s cpu \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --cudabacktrace=true \
  --cudabacktrace-threshold=0 \
  --gpu-metrics-device=all \
  --stats=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --capture-range-op=both \
  --multi-gpu=all \
  python3 $SCRIPT

echo "HTA profiling completed. Check $(basename $SCRIPT)_hta.nsys-rep for results."
