#!/bin/bash

# Master Profiling Script
# Runs all profiling tools for comprehensive performance analysis
# Supports Hopper H100/H200 and Blackwell B200/B300

set -e

SCRIPT_NAME="$1"
ARCH="${2:-auto}"
PROFILE_TOOLS="${3:-all}"  # all, nsys, ncu, hta, perf, pytorch, comprehensive
PROFILE_DURATION="${4:-30}"  # Duration in seconds

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

echo "=== Master Profiling for $SCRIPT_NAME ==="
echo "Architecture: $ARCH"
echo "Profile Tools: $PROFILE_TOOLS"
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
MASTER_PROFILE_DIR="master_profile_${ARCH}_${TIMESTAMP}"
mkdir -p "$MASTER_PROFILE_DIR"

echo "Creating master profile directory: $MASTER_PROFILE_DIR"
cd "$MASTER_PROFILE_DIR"

# Function to run specific profiling tool
run_profiling_tool() {
    local tool="$1"
    local tool_dir="profile_${tool}_${ARCH}"
    
    echo "Running $tool profiling..."
    mkdir -p "$tool_dir"
    cd "$tool_dir"
    
    case "$tool" in
        "nsys")
            bash ../../nsys_profile.sh "../../$SCRIPT_NAME" "$ARCH"
            ;;
        "ncu")
            bash ../../ncu_profile.sh "../../$SCRIPT_NAME" "$ARCH"
            ;;
        "hta")
            bash ../../hta_profile.sh "../../$SCRIPT_NAME" "$ARCH"
            ;;
        "perf")
            bash ../../perf_profile.sh "../../$SCRIPT_NAME" "$ARCH"
            ;;
        "pytorch")
            bash ../../pytorch_profile.sh "../../$SCRIPT_NAME" "$ARCH" "full"
            ;;
        "comprehensive")
            bash ../../comprehensive_profile.sh "../../$SCRIPT_NAME" "$ARCH" "$PROFILE_DURATION"
            ;;
        *)
            echo "Unknown profiling tool: $tool"
            return 1
            ;;
    esac
    
    cd ..
}

# Run selected profiling tools
if [ "$PROFILE_TOOLS" = "all" ]; then
    # Run all tools in parallel
    echo "Running all profiling tools in parallel..."
    
    # Start all profiling tools
    for tool in nsys ncu hta pytorch comprehensive; do
        run_profiling_tool "$tool" &
        TOOL_PIDS+=($!)
    done
    
    # Wait for all tools to complete
    echo "Waiting for all profiling tools to complete..."
    for pid in "${TOOL_PIDS[@]}"; do
        wait $pid
    done
    
    # Run perf separately (it needs to monitor the running process)
    echo "Running perf profiling..."
    run_profiling_tool "perf"
    
else
    # Run specific tools
    IFS=',' read -ra TOOLS <<< "$PROFILE_TOOLS"
    for tool in "${TOOLS[@]}"; do
        run_profiling_tool "$tool"
    done
fi

# Generate master report
echo "Generating master report..."
cat > "master_report_${ARCH}.md" << EOF
# Master Performance Analysis Report

## Test Information
- **Script**: $SCRIPT_NAME
- **Architecture**: $ARCH
- **Profile Tools**: $PROFILE_TOOLS
- **Duration**: $PROFILE_DURATION seconds
- **Timestamp**: $TIMESTAMP

## Architecture Details
EOF

if [ "$ARCH" = "sm_90" ]; then
    cat >> "master_report_${ARCH}.md" << EOF
- **GPU**: Hopper H100/H200
- **Compute Capability**: 9.0
- **Memory**: HBM3 (3.35 TB/s)
- **Features**: Transformer Engine, Dynamic Programming
- **Tensor Cores**: 4th Generation
EOF
elif [ "$ARCH" = "sm_100" ]; then
    cat >> "master_report_${ARCH}.md" << EOF
- **GPU**: Blackwell B200/B300
- **Compute Capability**: 10.0
- **Memory**: HBM3e (3.2 TB/s)
- **Features**: TMA, NVLink-C2C, Stream-ordered Memory
- **Tensor Cores**: 4th Generation
EOF
fi

cat >> "master_report_${ARCH}.md" << EOF

## Profiling Tools Used

### 1. Nsight Systems (nsys)
- **Purpose**: System-level timeline analysis
- **Directory**: profile_nsys_${ARCH}/
- **Files**: nsys_timeline_${ARCH}.nsys-rep
- **Analysis**: CUDA, NVTX, Python sampling, GPU metrics

### 2. Nsight Compute (ncu)
- **Purpose**: Kernel-level performance analysis
- **Directory**: profile_ncu_${ARCH}/
- **Files**: ncu_kernel_${ARCH}.ncu-rep
- **Analysis**: Occupancy, efficiency, memory throughput

### 3. HTA (Holistic Tracing Analysis)
- **Purpose**: Multi-GPU and distributed analysis
- **Directory**: profile_hta_${ARCH}/
- **Files**: hta_analysis_${ARCH}.nsys-rep
- **Analysis**: NCCL, multi-GPU communication

### 4. Perf
- **Purpose**: System-level CPU analysis
- **Directory**: profile_perf_${ARCH}/
- **Files**: perf_system_${ARCH}.data
- **Analysis**: CPU utilization, call graphs

### 5. PyTorch Profiler
- **Purpose**: Framework-level analysis
- **Directory**: profile_pytorch_${ARCH}/
- **Files**: chrome_trace_full.json, profiler_summary_full.txt
- **Analysis**: Memory, FLOPs, module breakdown

### 6. Comprehensive Profiling
- **Purpose**: All tools combined
- **Directory**: profile_comprehensive_${ARCH}/
- **Files**: Multiple analysis files
- **Analysis**: Complete performance picture

## Performance Recommendations

### For Hopper H100/H200 (SM90):
- **Transformer Engine**: Enable for transformer models
- **Dynamic Programming**: Use for variable workloads
- **HBM3 Optimization**: Maximize 3.35 TB/s bandwidth
- **Tensor Cores**: Leverage 4th generation for matrix ops
- **torch.compile**: Use with max-autotune mode

### For Blackwell B200/B300 (SM100):
- **TMA**: Use Tensor Memory Accelerator for efficient data movement
- **Stream-ordered Memory**: Use cudaMallocAsync/cudaFreeAsync
- **HBM3e**: Optimize for 3.2 TB/s bandwidth
- **NVLink-C2C**: Direct GPU-to-GPU communication
- **Blackwell Optimizations**: Enable architecture-specific features

## Analysis Workflow

### 1. Quick Overview
\`\`\`bash
# Check GPU utilization
nvidia-smi

# View PyTorch summary
cat profile_pytorch_${ARCH}/profiler_summary_full.txt
\`\`\`

### 2. Timeline Analysis
\`\`\`bash
# Open Nsight Systems
nsys-ui profile_nsys_${ARCH}/nsys_timeline_${ARCH}.nsys-rep
\`\`\`

### 3. Kernel Analysis
\`\`\`bash
# Open Nsight Compute
ncu-ui profile_ncu_${ARCH}/ncu_kernel_${ARCH}.ncu-rep
\`\`\`

### 4. Memory Analysis
\`\`\`bash
# Open memory profiling
nsys-ui profile_comprehensive_${ARCH}/memory_profile_${ARCH}.nsys-rep
\`\`\`

### 5. Framework Analysis
\`\`\`bash
# Open Chrome trace
# Navigate to chrome://tracing/ and load:
# profile_pytorch_${ARCH}/chrome_trace_full.json
\`\`\`

## Key Metrics to Monitor

### Performance Metrics
- **GPU Utilization**: Target >90%
- **Memory Bandwidth**: HBM3/HBM3e utilization
- **Tensor Core Usage**: For matrix operations
- **Kernel Occupancy**: Maximize thread block efficiency
- **Memory Latency**: Minimize access delays

### Architecture-Specific Metrics
- **Hopper**: Transformer Engine efficiency, dynamic programming usage
- **Blackwell**: TMA efficiency, stream-ordered memory usage, NVLink-C2C bandwidth

## Optimization Checklist

### General Optimizations
- [ ] High GPU utilization (>90%)
- [ ] Efficient memory access patterns
- [ ] Optimal kernel occupancy
- [ ] Minimized communication overhead
- [ ] Balanced workload distribution

### Architecture-Specific Optimizations
- [ ] **Hopper**: Transformer Engine enabled, HBM3 optimized
- [ ] **Blackwell**: TMA enabled, HBM3e optimized, stream-ordered memory

### Framework Optimizations
- [ ] torch.compile with max-autotune
- [ ] Dynamic shapes enabled
- [ ] Mixed precision training
- [ ] Memory-efficient training
- [ ] Distributed training optimized

## Next Steps
1. Review timeline analysis for bottlenecks
2. Analyze kernel performance for optimization opportunities
3. Check memory patterns for bandwidth optimization
4. Verify framework-level optimizations
5. Apply architecture-specific features
6. Re-run profiling to measure improvements

## Files Generated
- **Master Report**: master_report_${ARCH}.md
- **Nsight Systems**: profile_nsys_${ARCH}/nsys_timeline_${ARCH}.nsys-rep
- **Nsight Compute**: profile_ncu_${ARCH}/ncu_kernel_${ARCH}.ncu-rep
- **HTA Analysis**: profile_hta_${ARCH}/hta_analysis_${ARCH}.nsys-rep
- **Perf Data**: profile_perf_${ARCH}/perf_system_${ARCH}.data
- **PyTorch Trace**: profile_pytorch_${ARCH}/chrome_trace_full.json
- **Comprehensive**: profile_comprehensive_${ARCH}/comprehensive_report_${ARCH}.md
EOF

echo ""
echo "=== Master Profiling Complete ==="
echo "Results saved in: $MASTER_PROFILE_DIR"
echo "Master report: $MASTER_PROFILE_DIR/master_report_${ARCH}.md"
echo ""
echo "Quick start:"
echo "  cd $MASTER_PROFILE_DIR"
echo "  cat master_report_${ARCH}.md"
echo ""
echo "View results:"
echo "  nsys-ui profile_nsys_${ARCH}/nsys_timeline_${ARCH}.nsys-rep"
echo "  ncu-ui profile_ncu_${ARCH}/ncu_kernel_${ARCH}.ncu-rep"
echo "  # Open chrome://tracing/ and load profile_pytorch_${ARCH}/chrome_trace_full.json"
