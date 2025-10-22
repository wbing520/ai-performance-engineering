#!/bin/bash
# Profiling script for Chapter 10: Tensor Cores & Clusters
# Tests tcgen05 Tensor Cores, Thread Block Clusters, and TMA on Blackwell B200

set -e

echo "Chapter 10: Tensor Cores & Clusters Profiling"
echo "============================================="
echo ""

# Check if executables exist
if [ ! -f "./tcgen05_blackwell" ]; then
    echo "Building executables..."
    make
fi

# 1. Profile tcgen05 Tensor Cores
echo "1. Profiling tcgen05 Tensor Cores..."
echo "-------------------------------------"

nsys profile -o ch10_tcgen05_timeline \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    ./tcgen05_blackwell

ncu --set full \
    --target-processes all \
    --metrics sm__inst_executed_pipe_tensor.sum,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_tensor_op.sum \
    --export ch10_tcgen05_metrics \
    --force-overwrite \
    ./tcgen05_blackwell

echo "  Timeline: ch10_tcgen05_timeline.nsys-rep"
echo "  Metrics: ch10_tcgen05_metrics.ncu-rep"
echo ""

# 2. Profile Thread Block Clusters (8 CTAs)
echo "2. Profiling Thread Block Clusters..."
echo "--------------------------------------"

nsys profile -o ch10_clusters_timeline \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    ./cluster_group_blackwell

ncu --set full \
    --target-processes all \
    --metrics launch__cluster_size,\
launch__occupancy_per_cluster,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    --export ch10_clusters_metrics \
    --force-overwrite \
    ./cluster_group_blackwell

echo "  Timeline: ch10_clusters_timeline.nsys-rep"
echo "  Metrics: ch10_clusters_metrics.ncu-rep"
echo ""

# 3. Profile TMA (Tensor Memory Accelerator)
echo "3. Profiling TMA Pipeline..."
echo "----------------------------"

nsys profile -o ch10_tma_timeline \
    --trace=cuda,nvtx \
    --force-overwrite=true \
    ./tma_2d_pipeline_blackwell

ncu --set full \
    --target-processes all \
    --metrics l2_comp_read_throughput,\
l2_comp_write_throughput,\
dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --export ch10_tma_metrics \
    --force-overwrite \
    ./tma_2d_pipeline_blackwell

echo "  Timeline: ch10_tma_timeline.nsys-rep"
echo "  Metrics: ch10_tma_metrics.ncu-rep"
echo ""

# 4. Key metrics summary
echo "Key Metrics to Verify:"
echo "  tcgen05 Tensor Cores:"
echo "    - FP8 TFLOPS: Target >1200 TFLOPS"
echo "    - FP16 TFLOPS: Target >800 TFLOPS"
echo "    - Tensor Core utilization: Target >80%"
echo ""
echo "  Thread Block Clusters:"
echo "    - Cluster size: 8 CTAs (vs 4 on Hopper)"
echo "    - DSMEM usage: Up to 2 MB"
echo "    - SM utilization: Target >80% of 148 SMs"
echo ""
echo "  TMA:"
echo "    - Memory bandwidth: Target >7 TB/s"
echo "    - L2 cache hit rate: Target >70%"
echo ""

echo "Profiling complete!"
echo ""
echo "To analyze:"
echo "  nsys-ui ch10_*_timeline.nsys-rep"
echo "  ncu-ui ch10_*_metrics.ncu-rep"

