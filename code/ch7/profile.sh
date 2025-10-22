#!/bin/bash
# Profiling script for Chapter 7: Memory Optimization
# Tests HBM3e bandwidth optimizations on Blackwell B200

set -e

echo "Chapter 7: Memory Optimization Profiling"
echo "========================================"
echo ""

# Check if executables exist
if [ ! -f "./hbm3e_peak_bandwidth" ]; then
    echo "Building executables..."
    make
fi

# 1. Nsight Systems - Memory transfer timeline
echo "1. Running Nsight Systems profiling..."
nsys profile -o ch7_hbm3e_timeline \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --force-overwrite=true \
    ./hbm3e_peak_bandwidth

echo ""
echo "Timeline saved to: ch7_hbm3e_timeline.nsys-rep"
echo ""

# 2. Nsight Compute - Detailed memory metrics
echo "2. Running Nsight Compute profiling..."
ncu --set full \
    --target-processes all \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct \
    --export ch7_hbm3e_metrics \
    --force-overwrite \
    ./hbm3e_peak_bandwidth

echo ""
echo "Metrics saved to: ch7_hbm3e_metrics.ncu-rep"
echo ""

# 3. Key metrics to check
echo "Key Metrics to Verify:"
echo "  - HBM3e bandwidth: Target >7.0 TB/s (90%+ utilization)"
echo "  - Memory coalescing: Target >90%"
echo "  - Cache hit rates"
echo "  - Achieved bandwidth progression:"
echo "    * Standard: ~3.2 TB/s (42%)"
echo "    * Vectorized (float4): ~3.6 TB/s (46%)"
echo "    * HBM3e optimized: >7.0 TB/s (90%+)"
echo ""

echo "Expected output shows progression of optimizations:"
echo "  1. Baseline copy"
echo "  2. Vectorized copy (float4)"
echo "  3. HBM3e optimized (256-byte bursts + cache streaming)"
echo ""

echo "Profiling complete!"
echo ""
echo "To analyze:"
echo "  nsys-ui ch7_hbm3e_timeline.nsys-rep"
echo "  ncu-ui ch7_hbm3e_metrics.ncu-rep"

