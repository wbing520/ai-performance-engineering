#!/bin/bash

# Comprehensive Profiling Script
# Combines all profiling tools for maximum performance analysis
# Supports Hopper H100/H200 and Blackwell B200/B300
# Updated for PyTorch 2.8, CUDA 12.8, and Triton 3.3

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
echo "PyTorch 2.8, CUDA 12.8, Triton 3.3 Support"
echo ""

# Set environment variables for optimal profiling
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export TORCH_CUDNN_V8_API_ENABLED=1

# Enhanced environment variables for latest features
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Create timestamp for this profiling session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="comprehensive_profile_${ARCH}_${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"

echo "Creating profile directory: $PROFILE_DIR"
cd "$PROFILE_DIR"

# 1. Nsight Systems Timeline Analysis (Enhanced for latest features)
echo "1. Running Nsight Systems timeline analysis..."
nsys profile \
    --force-overwrite=true \
    -o "nsys_timeline_${ARCH}" \
    -t cuda,nvtx,osrt,cudnn,cublas,nccl,triton \
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
NSYS_PID=$!

# 2. Nsight Compute Kernel Analysis (Enhanced metrics)
echo "2. Running Nsight Compute kernel analysis..."
ncu \
    --mode=launch \
    --target-processes=python3 \
    --set full \
    --kernel-regex ".*" \
    --sampling-interval 1 \
    --sampling-max-passes 5 \
    --sampling-period 1000000 \
    --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput,sm__cycles_elapsed.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.pct_of_peak_sustained_elapsed,sm__cycles_elapsed.avg.pct_of_peak_sustained_elapsed \
    --export csv \
    -o "ncu_kernel_${ARCH}" \
    python "../$SCRIPT_NAME" &
NCU_PID=$!

# 3. Memory Profiling (Enhanced for HBM3/HBM3e)
echo "3. Running memory profiling..."
nsys profile \
    --force-overwrite=true \
    -o "memory_profile_${ARCH}" \
    -t cuda,cudamemcpy \
    --duration=$PROFILE_DURATION \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --capture-range-op=both \
    python "../$SCRIPT_NAME" &
MEMORY_PID=$!

# 4. HTA (Holistic Tracing Analysis) - Enhanced for multi-GPU
echo "4. Running HTA analysis..."
nsys profile \
    --force-overwrite=true \
    -o "hta_analysis_${ARCH}" \
    -t cuda,nvtx,osrt,cudnn,cublas,nccl,triton \
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

# 5. Perf System-Level Analysis (Enhanced)
echo "5. Running Perf system-level analysis..."
perf record \
    -g \
    -p $(pgrep python) \
    -o "perf_system_${ARCH}" \
    --call-graph=dwarf \
    --freq=1000 \
    -- sleep $PROFILE_DURATION &
PERF_PID=$!

# 6. GPU Metrics Monitoring (Enhanced)
echo "6. Monitoring GPU metrics..."
nvidia-smi dmon -s pucvmet -d 1 -o T > "gpu_metrics_${ARCH}.log" &
GPU_MONITOR_PID=$!

# 7. PyTorch Profiler (Enhanced for 2.8)
echo "7. Running PyTorch profiler..."
python -c "
import sys
sys.path.append('..')
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import time

# Configure architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f'{device_props.major}.{device_props.minor}'
    
    if compute_capability == '9.0':  # Hopper
        torch._inductor.config.triton.use_hopper_optimizations = True
        torch._inductor.config.triton.hbm3_optimizations = True
    elif compute_capability == '10.0':  # Blackwell
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True

# Run with enhanced profiler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
    profile_memory=True,
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
) as prof:
    # Import and run the target script
    import subprocess
    import sys
    subprocess.run([sys.executable, '../$SCRIPT_NAME'], timeout=$PROFILE_DURATION)

# Export results
prof.export_chrome_trace('pytorch_trace_${ARCH}.json')
with open('pytorch_summary_${ARCH}.txt', 'w') as f:
    f.write('PyTorch Profiler Summary\n')
    f.write('='*40 + '\n')
    f.write(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
" &
PYTORCH_PID=$!

# 8. Triton Profiler (if available)
echo "8. Running Triton profiler..."
python -c "
import sys
sys.path.append('..')
try:
    import triton
    print(f'Triton version: {triton.__version__}')
    
    # Run with Triton profiling
    import subprocess
    import sys
    subprocess.run([sys.executable, '../$SCRIPT_NAME'], timeout=$PROFILE_DURATION)
except ImportError:
    print('Triton not available')
" &
TRITON_PID=$!

# Wait for all profiling to complete
echo "Waiting for profiling to complete..."
wait $NSYS_PID $NCU_PID $MEMORY_PID $HTA_PID $PERF_PID $PYTORCH_PID
kill $GPU_MONITOR_PID 2>/dev/null || true
kill $TRITON_PID 2>/dev/null || true

# Generate comprehensive report
echo "Generating comprehensive report..."
cat > "comprehensive_report_${ARCH}.md" << EOF
# Comprehensive Performance Analysis Report

## Test Information
- **Script**: $SCRIPT_NAME
- **Architecture**: $ARCH
- **Timestamp**: $TIMESTAMP
- **Duration**: $PROFILE_DURATION seconds
- **PyTorch**: 2.8
- **CUDA**: 12.8
- **Triton**: 3.3

## Architecture Details
EOF

if [ "$ARCH" = "sm_90" ]; then
    cat >> "comprehensive_report_${ARCH}.md" << EOF
- **GPU**: Hopper H100/H200
- **Compute Capability**: 9.0
- **Memory**: HBM3
- **Features**: Transformer Engine, Dynamic Programming
- **Optimizations**: HBM3 optimizations, Hopper-specific kernels
EOF
elif [ "$ARCH" = "sm_100" ]; then
    cat >> "comprehensive_report_${ARCH}.md" << EOF
- **GPU**: Blackwell B200/B300
- **Compute Capability**: 10.0
- **Memory**: HBM3e
- **Features**: TMA, NVLink-C2C, Stream-ordered Memory
- **Optimizations**: HBM3e optimizations, Blackwell-specific kernels
EOF
fi

cat >> "comprehensive_report_${ARCH}.md" << EOF

## Profiling Results

### 1. Nsight Systems Timeline
- **File**: nsys_timeline_${ARCH}.nsys-rep
- **Analysis**: System-level timeline with CUDA, NVTX, Triton, and Python sampling
- **View**: nsys-ui nsys_timeline_${ARCH}.nsys-rep

### 2. Nsight Compute Kernel Analysis
- **File**: ncu_kernel_${ARCH}.ncu-rep
- **Analysis**: Detailed kernel-level performance metrics with enhanced metrics
- **View**: ncu-ui ncu_kernel_${ARCH}.ncu-rep

### 3. Memory Profiling
- **File**: memory_profile_${ARCH}.nsys-rep
- **Analysis**: Memory allocation and transfer patterns (HBM3/HBM3e optimized)
- **View**: nsys-ui memory_profile_${ARCH}.nsys-rep

### 4. HTA Analysis
- **File**: hta_analysis_${ARCH}.nsys-rep
- **Analysis**: Holistic tracing for multi-GPU systems with Triton support
- **View**: nsys-ui hta_analysis_${ARCH}.nsys-rep

### 5. Perf System Analysis
- **File**: perf_system_${ARCH}.data
- **Analysis**: System-level CPU and call graph analysis with enhanced sampling
- **View**: perf report -i perf_system_${ARCH}.data

### 6. GPU Metrics
- **File**: gpu_metrics_${ARCH}.log
- **Analysis**: Real-time GPU utilization and memory usage

### 7. PyTorch Profiler
- **File**: pytorch_trace_${ARCH}.json
- **Analysis**: Framework-level profiling with PyTorch 2.8 features
- **View**: chrome://tracing/ (load pytorch_trace_${ARCH}.json)

### 8. Triton Profiler
- **Analysis**: Triton 3.3 kernel profiling and optimization

## Performance Recommendations

### For Hopper H100/H200 (SM90):
- Enable Transformer Engine optimizations
- Use dynamic programming features
- Optimize for HBM3 memory bandwidth
- Leverage 4th generation Tensor Cores
- Use TMA for efficient memory transfers
- Enable Hopper-specific Triton optimizations

### For Blackwell B200/B300 (SM100):
- Enable TMA (Tensor Memory Accelerator)
- Use stream-ordered memory allocation
- Optimize for HBM3e memory bandwidth
- Leverage NVLink-C2C for GPU communication
- Use Blackwell-specific Triton kernels
- Enable HBM3e memory optimizations

## Latest Features Used
- **PyTorch 2.8**: Enhanced compiler, dynamic shapes, improved profiler
- **CUDA 12.8**: Latest CUDA features, improved kernel performance
- **Triton 3.3**: Latest Triton optimizations, architecture-specific kernels
- **Enhanced Profiling**: Nsight Systems 2025.1, Nsight Compute 2025.1
- **HTA**: Holistic Tracing Analysis for multi-GPU systems
- **Perf**: Enhanced system-level analysis

## Next Steps
1. Open Nsight Systems UI: nsys-ui nsys_timeline_${ARCH}.nsys-rep
2. Open Nsight Compute UI: ncu-ui ncu_kernel_${ARCH}.ncu-rep
3. Analyze memory patterns: nsys-ui memory_profile_${ARCH}.nsys-rep
4. Review system performance: perf report -i perf_system_${ARCH}.data
5. Check GPU metrics: cat gpu_metrics_${ARCH}.log
6. View PyTorch trace: chrome://tracing/ → Load pytorch_trace_${ARCH}.json

## Optimization Opportunities
- Identify kernel bottlenecks
- Analyze memory access patterns
- Check for communication overhead
- Verify occupancy and utilization
- Look for pipeline stalls
- Optimize for architecture-specific features
- Use latest profiling tools for detailed analysis
EOF

echo "Comprehensive profiling completed!"
echo "Results saved in: $PROFILE_DIR"
echo "Report generated: $PROFILE_DIR/comprehensive_report_${ARCH}.md"
echo ""
echo "To view results:"
echo "  cd $PROFILE_DIR"
echo "  nsys-ui nsys_timeline_${ARCH}.nsys-rep"
echo "  ncu-ui ncu_kernel_${ARCH}.ncu-rep"
echo "  # Open chrome://tracing/ and load pytorch_trace_${ARCH}.json"
