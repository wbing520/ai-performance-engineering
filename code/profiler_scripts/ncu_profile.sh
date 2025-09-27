#!/bin/bash

# Nsight Compute Profiling Script
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

echo "Nsight Compute profiling for $SCRIPT_NAME (Architecture: $ARCH)"

ncu \
    --mode=launch \
    --target-processes=python3 \
    --set full \
    --clock-control none \
    --kernel-name regex:.* \
    --sampling-interval 1 \
    --sampling-max-passes 5 \
    --sampling-period 1000000 \
    -o "ncu_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    python "$SCRIPT_NAME"
