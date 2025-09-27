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

BASE_METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_average_branch_divergence.pct,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,shared_load_sectors,shared_store_sectors"

if [[ -n "${NCU_EXTRA_METRICS:-}" ]]; then
    METRIC_LIST="${BASE_METRICS},${NCU_EXTRA_METRICS}"
else
    METRIC_LIST="$BASE_METRICS"
fi

ncu \
    --set full \
    --clock-control none \
    --kernel-name regex:.* \
    --metrics "$METRIC_LIST" \
    --import-source yes \
    -o "ncu_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    python "$SCRIPT_NAME"
