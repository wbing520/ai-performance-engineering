#!/bin/bash

# HTA (Holistic Tracing Analysis) Profiling Script
# Supports Hopper H100/H200 and Blackwell B200/B300

SCRIPT_NAME="$1"
ARCH="${2:-auto}"

# Auto-detect architecture if not specified
if [ "$ARCH" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if [[ "$gpu_name" == *"H100"* ]] || [[ "$gpu_name" == *"H200"* ]]; then
            ARCH="sm_90"
        elif [[ "$gpu_name" == *"B200"* ]] || [[ "$gpu_name" == *"B300"* ]]; then
            ARCH="sm_100"
        else
            ARCH="sm_90"
        fi
    else
        ARCH="sm_90"
    fi
fi

echo "HTA profiling for $SCRIPT_NAME (Architecture: $ARCH)"

nsys profile \
    --force-overwrite=true \
    -o "hta_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
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
    python "$SCRIPT_NAME"
