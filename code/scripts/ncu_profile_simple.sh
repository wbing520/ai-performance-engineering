#!/bin/bash

# Simplified Nsight Compute Profiling Script for PyTorch Applications
# Targets Blackwell B200/B300 (SM100) with minimal metrics to avoid hanging

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
SCRIPT_NAME="$1"
ARCH="${2:-sm_100}"

echo "Simplified Nsight Compute profiling for $SCRIPT_NAME (Architecture: $ARCH)"

# Use minimal metrics to avoid hanging
# Profile only essential performance metrics
ncu \
    --set full \
    --clock-control none \
    --kernel-name regex:"vectorized_gather_kernel" \
    --metrics "sm__warps_active.avg.pct_of_peak_sustained_active" \
    --import-source no \
    -o "ncu_simple_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
    python3 "$SCRIPT_NAME"
