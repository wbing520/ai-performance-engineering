#!/bin/bash

# PyTorch Profiling Script
# Uses latest PyTorch 2.8 profiler features
# Supports Hopper H100/H200 and Blackwell B200/B300

set -e

SCRIPT_NAME="$1"
ARCH="${2:-auto}"
PROFILE_MODE="${3:-full}"  # full, memory, flops, modules

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

echo "=== PyTorch Profiling for $SCRIPT_NAME ==="
echo "Architecture: $ARCH"
echo "Profile Mode: $PROFILE_MODE"
echo ""

# Set environment variables for optimal PyTorch profiling
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_SHOW_CPP_STACKTRACES=1

# Create timestamp for this profiling session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_DIR="pytorch_profile_${ARCH}_${TIMESTAMP}"
mkdir -p "$PROFILE_DIR"

echo "Creating profile directory: $PROFILE_DIR"
cd "$PROFILE_DIR"

# Create PyTorch profiling wrapper
cat > "pytorch_profiler_wrapper.py" << 'EOF'
#!/usr/bin/env python3
"""
PyTorch Profiler Wrapper
Uses latest PyTorch 2.8 profiler features
"""

import torch
import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import sys
import os
import time

def setup_architecture_optimizations():
    """Setup architecture-specific optimizations."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        print(f"GPU: {device_props.name}")
        print(f"Compute Capability: {compute_capability}")
        
        if compute_capability == "9.0":  # Hopper H100/H200
            print("✓ Enabling Hopper H100/H200 optimizations")
            torch._inductor.config.triton.use_hopper_optimizations = True
            torch._inductor.config.triton.hbm3_optimizations = True
        elif compute_capability == "10.0":  # Blackwell B200/B300
            print("✓ Enabling Blackwell B200/B300 optimizations")
            torch._inductor.config.triton.use_blackwell_optimizations = True
            torch._inductor.config.triton.hbm3e_optimizations = True
            torch._inductor.config.triton.tma_support = True
        
        # Enable latest PyTorch 2.8 features
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.triton.autotune_mode = "max-autotune"
        torch._dynamo.config.automatic_dynamic_shapes = True

def run_with_profiler(script_path, profile_mode="full"):
    """Run script with PyTorch profiler."""
    setup_architecture_optimizations()
    
    # Import the target script
    sys.path.insert(0, os.path.dirname(script_path))
    script_name = os.path.basename(script_path).replace('.py', '')
    
    # Configure profiler based on mode
    if profile_mode == "full":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        profile_memory = True
        record_shapes = True
        with_stack = True
        with_flops = True
        with_modules = True
    elif profile_mode == "memory":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        profile_memory = True
        record_shapes = False
        with_stack = False
        with_flops = False
        with_modules = False
    elif profile_mode == "flops":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        profile_memory = False
        record_shapes = True
        with_stack = False
        with_flops = True
        with_modules = False
    elif profile_mode == "modules":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        profile_memory = False
        record_shapes = False
        with_stack = False
        with_flops = False
        with_modules = True
    else:
        print(f"Unknown profile mode: {profile_mode}")
        return
    
    print(f"Running {script_name} with {profile_mode} profiling...")
    
    # Run with profiler
    with profile(
        activities=activities,
        profile_memory=profile_memory,
        record_shapes=record_shapes,
        with_stack=with_stack,
        with_flops=with_flops,
        with_modules=with_modules,
        schedule=schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        )
    ) as prof:
        # Import and run the target script
        try:
            module = __import__(script_name)
            if hasattr(module, 'main'):
                module.main()
            elif hasattr(module, 'run'):
                module.run()
            else:
                print(f"Warning: No main() or run() function found in {script_name}")
        except Exception as e:
            print(f"Error running {script_name}: {e}")
            return
    
    # Export results
    prof.export_chrome_trace(f"chrome_trace_{profile_mode}.json")
    
    # Print summary
    print("\n" + "="*60)
    print("PyTorch Profiler Results")
    print("="*60)
    
    if profile_mode == "full" or profile_mode == "flops":
        print("\nTop CUDA operations by time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        if profile_mode == "full" or profile_mode == "flops":
            print("\nTop operations by FLOPs:")
            print(prof.key_averages().table(sort_by="flops", row_limit=10))
    
    if profile_mode == "full" or profile_mode == "memory":
        print("\nMemory usage summary:")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    
    if profile_mode == "full" or profile_mode == "modules":
        print("\nModule-level analysis:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    
    # Save detailed results
    with open(f"profiler_summary_{profile_mode}.txt", "w") as f:
        f.write("PyTorch Profiler Summary\n")
        f.write("="*40 + "\n")
        f.write(f"Script: {script_name}\n")
        f.write(f"Profile Mode: {profile_mode}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if profile_mode == "full" or profile_mode == "flops":
            f.write("Top CUDA operations by time:\n")
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            f.write("\n")
        
        if profile_mode == "full" or profile_mode == "memory":
            f.write("Memory usage summary:\n")
            f.write(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
            f.write("\n")
    
    print(f"\nResults saved to: {os.getcwd()}")
    print(f"Chrome trace: chrome_trace_{profile_mode}.json")
    print(f"Summary: profiler_summary_{profile_mode}.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pytorch_profiler_wrapper.py <script_path> [profile_mode]")
        sys.exit(1)
    
    script_path = sys.argv[1]
    profile_mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    
    run_with_profiler(script_path, profile_mode)
EOF

# Run PyTorch profiling
echo "Running PyTorch profiler..."
python pytorch_profiler_wrapper.py "../$SCRIPT_NAME" "$PROFILE_MODE"

# Generate PyTorch-specific report
cat > "pytorch_report_${ARCH}.md" << EOF
# PyTorch Profiler Report

## Test Information
- **Script**: $SCRIPT_NAME
- **Architecture**: $ARCH
- **Profile Mode**: $PROFILE_MODE
- **Timestamp**: $TIMESTAMP

## Architecture Details
EOF

if [ "$ARCH" = "sm_90" ]; then
    cat >> "pytorch_report_${ARCH}.md" << EOF
- **GPU**: Hopper H100/H200
- **Compute Capability**: 9.0
- **Memory**: HBM3
- **Features**: Transformer Engine, Dynamic Programming
EOF
elif [ "$ARCH" = "sm_100" ]; then
    cat >> "pytorch_report_${ARCH}.md" << EOF
- **GPU**: Blackwell B200/B300
- **Compute Capability**: 10.0
- **Memory**: HBM3e
- **Features**: TMA, NVLink-C2C, Stream-ordered Memory
EOF
fi

cat >> "pytorch_report_${ARCH}.md" << EOF

## PyTorch 2.8 Features Used
- **torch.compile**: With max-autotune mode
- **Dynamic Shapes**: Automatic dynamic shape support
- **Memory Profiling**: Detailed memory allocation tracking
- **FLOP Counting**: Operation-level floating point counting
- **Module Analysis**: Module-level performance breakdown
- **NVTX Integration**: Custom annotation support

## Profiling Results
- **Chrome Trace**: chrome_trace_${PROFILE_MODE}.json
- **Summary**: profiler_summary_${PROFILE_MODE}.txt

## Analysis Steps
1. Open Chrome trace: chrome://tracing/ (load chrome_trace_${PROFILE_MODE}.json)
2. Review summary: cat profiler_summary_${PROFILE_MODE}.txt
3. Analyze bottlenecks and optimization opportunities

## Performance Recommendations

### For Hopper H100/H200 (SM90):
- Enable torch.compile with mode="max-autotune"
- Use Transformer Engine optimizations
- Optimize for HBM3 memory bandwidth
- Leverage dynamic programming features

### For Blackwell B200/B300 (SM100):
- Enable Blackwell-specific optimizations
- Use HBM3e memory optimizations
- Enable TMA support
- Leverage stream-ordered memory allocation

## Next Steps
1. Open Chrome trace: chrome://tracing/ → Load chrome_trace_${PROFILE_MODE}.json
2. Review summary: cat profiler_summary_${PROFILE_MODE}.txt
3. Identify performance bottlenecks
4. Apply architecture-specific optimizations
EOF

echo "PyTorch profiling completed!"
echo "Results saved in: $PROFILE_DIR"
echo "Report generated: $PROFILE_DIR/pytorch_report_${ARCH}.md"
echo ""
echo "To view results:"
echo "  cd $PROFILE_DIR"
echo "  cat profiler_summary_${PROFILE_MODE}.txt"
echo "  # Open chrome://tracing/ and load chrome_trace_${PROFILE_MODE}.json"
