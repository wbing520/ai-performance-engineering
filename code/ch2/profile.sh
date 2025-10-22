#!/bin/bash
# Profiling script for Chapter 2: Hardware Architecture
# Tests NVLink-C2C bandwidth on Blackwell B200

set -e

echo "Chapter 2: Hardware Architecture Profiling"
echo "==========================================="
echo ""

# Check if executables exist
if [ ! -f "./nvlink_c2c_p2p_blackwell" ]; then
    echo "Building executables..."
    make
fi

# 1. Nsight Systems - Timeline view of NVLink transfers
echo "1. Running Nsight Systems profiling..."
nsys profile -o ch2_nvlink_timeline \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    ./nvlink_c2c_p2p_blackwell

echo ""
echo "Timeline saved to: ch2_nvlink_timeline.nsys-rep"
echo "Open with: nsys-ui ch2_nvlink_timeline.nsys-rep"
echo ""

# 2. Nsight Compute - Kernel-level metrics
echo "2. Running Nsight Compute profiling..."
ncu --set full \
    --target-processes all \
    --export ch2_nvlink_metrics \
    --force-overwrite \
    ./nvlink_c2c_p2p_blackwell

echo ""
echo "Metrics saved to: ch2_nvlink_metrics.ncu-rep"
echo "Open with: ncu-ui ch2_nvlink_metrics.ncu-rep"
echo ""

# 3. Key metrics to check
echo "Key Metrics to Verify:"
echo "  - NVLink-C2C bandwidth: Target ~900 GB/s"
echo "  - PCIe Gen5 bandwidth: Target ~64 GB/s"
echo "  - CPU-GPU coherent transfers"
echo ""

echo "Profiling complete!"

