#!/bin/bash

# Perf System-Level Profiling Script
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

echo "Perf profiling for $SCRIPT_NAME (Architecture: $ARCH)"

perf record \
    -g \
    -p $(pgrep python) \
    -o "perf_data_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    python "$SCRIPT_NAME"

perf report -i "perf_data_${ARCH}_$(date +%Y%m%d_%H%M%S)"
