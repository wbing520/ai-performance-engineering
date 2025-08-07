#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1

# Triton 3.4 profiler for kernel-level analysis
echo "Running Triton 3.4 profiler..."

# Set Triton environment variables for profiling
export TRITON_DEBUG=1
export TRITON_PROFILER=1
export TRITON_PROFILER_OUTPUT=triton_profile.json

# Run with Triton profiling enabled
python3 $SCRIPT

echo "Triton profiling completed. Check triton_profile.json for results."
