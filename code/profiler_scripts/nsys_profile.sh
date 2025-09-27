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

BASE_TRACE="cuda,nvtx,osrt,cublas,cudnn,nvlink"
TRACE_OPTS=(--force-overwrite=true -o "nsys_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" -t "$BASE_TRACE" -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true --cudabacktrace-threshold=0 --gpu-metrics-device=all --stats=true)

if [[ -n "${NSYS_EXTRA_OPTS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_OPTS=($NSYS_EXTRA_OPTS)
    TRACE_OPTS+=(${EXTRA_OPTS[@]})
fi

nsys profile "${TRACE_OPTS[@]}" python "$SCRIPT_NAME"
