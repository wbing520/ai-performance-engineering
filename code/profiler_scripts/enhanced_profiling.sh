#!/bin/bash

# Enhanced Profiling Script for Blackwell
# Targets Blackwell B200/B300 (SM100)

set -e

# Configuration
SCRIPT_NAME="$1"
ARCH="${2:-auto}"  # auto or sm_100
PROFILE_TYPE="${3:-all}"  # nsys, ncu, hta, perf, all

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

echo "Enhanced Profiling for $SCRIPT_NAME"
echo "Architecture: $ARCH"
echo "Profile Type: $PROFILE_TYPE"

# Set environment variables for optimal profiling
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Function to run Nsight Systems profiling
run_nsys() {
    echo "Running Nsight Systems timeline profiling..."
    nsys profile \
        --force-overwrite=true \
        -o "nsys_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
        -t cuda,nvtx,osrt,cudnn,cublas \
        -s cpu \
        --python-sampling=true \
        --python-sampling-frequency=1000 \
        --cudabacktrace=true \
        --cudabacktrace-threshold=0 \
        --gpu-metrics-device=all \
        --stats=true \
        python "$SCRIPT_NAME"
}

# Function to run Nsight Compute profiling
run_ncu() {
    echo "Running Nsight Compute kernel profiling..."
    ncu \
        --mode=launch \
        --target-processes=python3 \
        --set full \
        --clock-control none \
        --kernel-regex ".*" \
    --sampling-interval 1 \
    --sampling-max-passes 5 \
    --sampling-period 1000000 \
    -o "ncu_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
        python "$SCRIPT_NAME"
}

# Function to run HTA profiling
run_hta() {
    echo "Running HTA (Holistic Tracing Analysis) profiling..."
    nsys profile \
        --force-overwrite=true \
        -o "hta_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
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
        python "$SCRIPT_NAME"
}

# Function to run Perf profiling
run_perf() {
    echo "Running Perf system-level profiling..."
    perf record \
        -g \
        -p $(pgrep python) \
        -o "perf_data_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
        python "$SCRIPT_NAME"
    
    perf report -i "perf_data_${ARCH}_$(date +%Y%m%d_%H%M%S)"
}

# Function to run comprehensive profiling
run_comprehensive() {
    echo "Running comprehensive profiling with all tools..."
    
    # 1. Nsight Systems timeline
    echo "1. Nsight Systems timeline..."
    run_nsys
    
    # 2. Nsight Compute kernel analysis
    echo "2. Nsight Compute kernel analysis..."
    run_ncu
    
    # 3. Memory profiling
    echo "3. Memory profiling..."
    nsys profile \
        --force-overwrite=true \
        -o "memory_profile_${ARCH}_$(date +%Y%m%d_%H%M%S)" \
        -t cuda,cudamemcpy \
        python "$SCRIPT_NAME"
    
    # 4. HTA analysis
    echo "4. HTA analysis..."
    run_hta
    
    # 5. Perf analysis
    echo "5. Perf analysis..."
    run_perf
}

# Run profiling based on type
case "$PROFILE_TYPE" in
    "nsys")
        run_nsys
        ;;
    "ncu")
        run_ncu
        ;;
    "hta")
        run_hta
        ;;
    "perf")
        run_perf
        ;;
    "all")
        run_comprehensive
        ;;
    *)
        echo "Unknown profile type: $PROFILE_TYPE"
        echo "Available types: nsys, ncu, hta, perf, all"
        exit 1
        ;;
esac

echo "Profiling completed for $SCRIPT_NAME with $ARCH architecture"
