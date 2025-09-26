#!/bin/bash

# Nsight Systems Profiling Script
# Targets Blackwell B200/B300 (SM100)

SCRIPT_NAME="$1"
ARCH="${2:-auto}"

# Auto-detect architecture if not specified
if [ "$ARCH" = "auto" ]; then
    ARCH="sm_100"
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if [[ ! "$gpu_name" =~ B200|B300 ]]; then
            echo "⚠ Non-Blackwell GPU detected; running with sm_100 profile." >&2
        fi
    else
        echo "⚠ Unable to query GPU via nvidia-smi; assuming Blackwell profile." >&2
    fi
fi

echo "Nsight Systems profiling for $SCRIPT_NAME (Architecture: $ARCH)"

nsys profile \
    --force-overwrite=true \
    -o "nsys_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    -t cuda,nvtx,osrt,cudnn,cublas \
    -s cpu \
    --python-sampling=true \
    --python-sampling-frequency=1000 \
    --cudabacktrace=true \
    --cudabacktrace-threshold=0 \
    --gpu-metrics-device=all \
    --stats=true \
    python "$SCRIPT_NAME"
