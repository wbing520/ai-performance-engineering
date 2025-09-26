#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1

# Latest Nsight Compute profiling for CUDA 12.8 and Blackwell B200/B300
ncu \
  --mode=launch \
  --target-processes=python3 \
  --set full \
  --kernel-regex ".*" \
  --sampling-interval 1 \
  --sampling-max-passes 5 \
  --sampling-period 1000000 \
  -o $(basename $SCRIPT)_ncu \
  python3 $SCRIPT
