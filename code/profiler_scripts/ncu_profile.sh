#!/bin/bash

# Nsight Compute Profiling Script
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

echo "Nsight Compute profiling for $SCRIPT_NAME (Architecture: $ARCH)"

ncu \
    --mode=launch \
    --target-processes=python3 \
    --set full \
    --clock-control none \
    --kernel-regex ".*" \
    --sampling-interval 1 \
    --sampling-max-passes 5 \
    --sampling-period 1000000 \
    --export csv \
    -o "ncu_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    python "$SCRIPT_NAME"
