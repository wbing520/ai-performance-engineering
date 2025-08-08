#!/bin/bash

# Comprehensive Profiling Script
# Combines all profiling tools for maximum performance analysis
# Supports Hopper H100/H200 and Blackwell B200/B300

set -e

SCRIPT_NAME="$1"
ARCH="${2:-auto}"
PROFILE_DURATION="${3:-30}"  # Duration in seconds

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

echo "=== Comprehensive Profiling for $SCRIPT_NAME ==="
echo "Architecture: $ARCH"
echo "Duration: $PROFILE_DURATION seconds"
echo ""

# Set environment variables for optimal profiling
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export TORCH_CUDNN_V8_API_ENABLED=1

# Create timestamp for this profiling session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="comprehensive_profile_${ARCH}_${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"

echo "Creating profile directory: $PROFILE_DIR"
cd "$PROFILE_DIR"

# 1. Nsight Systems Timeline Analysis
echo "1. Running Nsight Systems timeline analysis..."
nsys profile \
    --force-overwrite=true \
    -o "nsys_timeline_${ARCH}" \
    -t cuda,nvtx,osrt,cudnn,cublas,nccl \
    -s cpu \
    --python-sampling=true \
    --python-sampling-frequency=1000 \
    --cudabacktrace=true \
    --cudabacktrace-threshold=0 \
    --gpu-metrics-device=all \
    --stats=true \
    --duration=$PROFILE_DURATION \
    python "../$SCRIPT_NAME" &
NSYS_PID=$!

# 2. Nsight Compute Kernel Analysis
echo "2. Running Nsight Compute kernel analysis..."
ncu \
    --mode=launch \
    --target-processes=python3 \
    --set full \
    --kernel-regex ".*" \
    --sampling-interval 1 \
    --sampling-max-passes 5 \
    --sampling-period 1000000 \
    --export csv \
    -o "ncu_kernel_${ARCH}" \
    python "../$SCRIPT_NAME" &
NCU_PID=$!

# 3. Memory Profiling
echo "3. Running memory profiling..."
nsys profile \
    --force-overwrite=true \
    -o "memory_profile_${ARCH}" \
    -t cuda,cudamemcpy \
    --duration=$PROFILE_DURATION \
    python "../$SCRIPT_NAME" &
MEMORY_PID=$!

# 4. HTA (Holistic Tracing Analysis)
echo "4. Running HTA analysis..."
nsys profile \
    --force-overwrite=true \
    -o "hta_analysis_${ARCH}" \
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
    --duration=$PROFILE_DURATION \
    python "../$SCRIPT_NAME" &
HTA_PID=$!

# 5. Perf System-Level Analysis
echo "5. Running Perf system-level analysis..."
perf record \
    -g \
    -p $(pgrep python) \
    -o "perf_system_${ARCH}" \
    -- sleep $PROFILE_DURATION &
PERF_PID=$!

# 6. GPU Metrics Monitoring
echo "6. Monitoring GPU metrics..."
nvidia-smi dmon -s pucvmet -d 1 -o T > "gpu_metrics_${ARCH}.log" &
GPU_MONITOR_PID=$!

# Wait for all profiling to complete
echo "Waiting for profiling to complete..."
wait $NSYS_PID $NCU_PID $MEMORY_PID $HTA_PID $PERF_PID
kill $GPU_MONITOR_PID 2>/dev/null || true

# Generate comprehensive report
echo "Generating comprehensive report..."
cat > "comprehensive_report_${ARCH}.md" << EOF
# Comprehensive Performance Analysis Report

## Test Information
- **Script**: $SCRIPT_NAME
- **Architecture**: $ARCH
- **Timestamp**: $TIMESTAMP
- **Duration**: $PROFILE_DURATION seconds

## Architecture Details
EOF

if [ "$ARCH" = "sm_90" ]; then
    cat >> "comprehensive_report_${ARCH}.md" << EOF
- **GPU**: Hopper H100/H200
- **Compute Capability**: 9.0
- **Memory**: HBM3
- **Features**: Transformer Engine, Dynamic Programming
EOF
elif [ "$ARCH" = "sm_100" ]; then
    cat >> "comprehensive_report_${ARCH}.md" << EOF
- **GPU**: Blackwell B200/B300
- **Compute Capability**: 10.0
- **Memory**: HBM3e
- **Features**: TMA, NVLink-C2C, Stream-ordered Memory
EOF
fi

cat >> "comprehensive_report_${ARCH}.md" << EOF

## Profiling Results

### 1. Nsight Systems Timeline
- **File**: nsys_timeline_${ARCH}.nsys-rep
- **Analysis**: System-level timeline with CUDA, NVTX, and Python sampling
- **View**: nsys-ui nsys_timeline_${ARCH}.nsys-rep

### 2. Nsight Compute Kernel Analysis
- **File**: ncu_kernel_${ARCH}.ncu-rep
- **Analysis**: Detailed kernel-level performance metrics
- **View**: ncu-ui ncu_kernel_${ARCH}.ncu-rep

### 3. Memory Profiling
- **File**: memory_profile_${ARCH}.nsys-rep
- **Analysis**: Memory allocation and transfer patterns
- **View**: nsys-ui memory_profile_${ARCH}.nsys-rep

### 4. HTA Analysis
- **File**: hta_analysis_${ARCH}.nsys-rep
- **Analysis**: Holistic tracing for multi-GPU systems
- **View**: nsys-ui hta_analysis_${ARCH}.nsys-rep

### 5. Perf System Analysis
- **File**: perf_system_${ARCH}.data
- **Analysis**: System-level CPU and call graph analysis
- **View**: perf report -i perf_system_${ARCH}.data

### 6. GPU Metrics
- **File**: gpu_metrics_${ARCH}.log
- **Analysis**: Real-time GPU utilization and memory usage

## Performance Recommendations

### For Hopper H100/H200 (SM90):
- Enable Transformer Engine optimizations
- Use dynamic programming features
- Optimize for HBM3 memory bandwidth
- Leverage 4th generation Tensor Cores

### For Blackwell B200/B300 (SM100):
- Enable TMA (Tensor Memory Accelerator)
- Use stream-ordered memory allocation
- Optimize for HBM3e memory bandwidth
- Leverage NVLink-C2C for GPU communication

## Next Steps
1. Open Nsight Systems UI: nsys-ui nsys_timeline_${ARCH}.nsys-rep
2. Open Nsight Compute UI: ncu-ui ncu_kernel_${ARCH}.ncu-rep
3. Analyze memory patterns: nsys-ui memory_profile_${ARCH}.nsys-rep
4. Review system performance: perf report -i perf_system_${ARCH}.data
5. Check GPU metrics: cat gpu_metrics_${ARCH}.log

## Optimization Opportunities
- Identify kernel bottlenecks
- Analyze memory access patterns
- Check for communication overhead
- Verify occupancy and utilization
- Look for pipeline stalls
EOF

echo "Comprehensive profiling completed!"
echo "Results saved in: $PROFILE_DIR"
echo "Report generated: $PROFILE_DIR/comprehensive_report_${ARCH}.md"
echo ""
echo "To view results:"
echo "  cd $PROFILE_DIR"
echo "  nsys-ui nsys_timeline_${ARCH}.nsys-rep"
echo "  ncu-ui ncu_kernel_${ARCH}.ncu-rep"
