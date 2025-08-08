# Performance Profiling Guide

## Overview
This guide covers the latest profiling tools and best practices for:
- **CUDA 12.4**
- **PyTorch 2.8**
- **OpenAI Triton 3.4**
- **Modern GPUs** (Ampere, Ada Lovelace, Hopper)

## Profiling Tools

### 1. Nsight Systems (nsys)
**Latest Version**: 2025.2.1
**Purpose**: System-level timeline analysis

```bash
# Basic profiling
bash profiler_scripts/nsys_profile.sh your_script.py

# Advanced profiling with Python sampling
nsys profile \
  --force-overwrite=true \
  -o profile_report \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s cpu \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --cudabacktrace=true \
  --cudabacktrace-threshold=0 \
  --gpu-metrics-device=all \
  --stats=true \
  python your_script.py
```

**Key Features for CUDA 12.4**:
- Python backtrace sampling
- CUDA backtrace integration
- Multi-GPU support
- Hardware metrics collection

### 2. Nsight Compute (ncu)
**Latest Version**: 2024.3
**Purpose**: Kernel-level performance analysis

```bash
# Basic profiling
bash profiler_scripts/ncu_profile.sh your_script.py

# Advanced profiling
ncu \
  --mode=launch \
  --target-processes=python3 \
  --set full \
  --kernel-regex ".*" \
  --sampling-interval 1 \
  --sampling-max-passes 5 \
  --sampling-period 1000000 \
  --export csv \
  -o ncu_report \
  python your_script.py
```

**Key Features for Modern GPUs**:
- Multi-architecture support (sm_80, sm_86, sm_90)
- Tensor Core metrics
- Memory bandwidth analysis
- Advanced occupancy analysis

### 3. PyTorch Profiler
**Latest Version**: PyTorch 2.8
**Purpose**: Framework-level profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx

# Latest PyTorch 2.8 profiler configuration
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True, 
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    with_modules=True,
    schedule=schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    )
) as prof:
    # Your training loop here
    pass

# Export results
prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

**Key Features for PyTorch 2.8**:
- Enhanced memory profiling
- FLOP counting
- Module-level analysis
- TensorBoard integration

### 4. Triton Profiler
**Latest Version**: Triton 3.4
**Purpose**: Kernel generation analysis

```bash
# Triton profiling
bash profiler_scripts/triton_profile.sh your_script.py

# Environment variables for Triton profiling
export TRITON_DEBUG=1
export TRITON_PROFILER=1
export TRITON_PROFILER_OUTPUT=triton_profile.json
```

**Key Features for Triton 3.4**:
- Kernel generation analysis
- Autotuning insights
- Memory access patterns
- Performance optimization suggestions

### 5. Holistic Tracing Analysis (HTA)
**Purpose**: Multi-GPU and distributed profiling

```bash
# HTA profiling
bash profiler_scripts/hta_profile.sh your_script.py

# Advanced HTA configuration
nsys profile \
  --force-overwrite=true \
  -o hta_report \
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
  python your_script.py
```

**Key Features for Multi-GPU**:
- NCCL communication analysis
- Cross-GPU synchronization
- Load balancing analysis
- Memory transfer optimization

### 6. Memory Profiler
**Purpose**: Memory usage and optimization analysis

```bash
# Memory profiling
bash profiler_scripts/memory_profile.sh your_script.py
```

**Key Features**:
- Memory allocation tracking
- Memory fragmentation analysis
- Peak memory usage
- Memory optimization suggestions

## Best Practices

### 1. Profiling Workflow
1. **Start with Nsight Systems**: Get system-level overview
2. **Use PyTorch Profiler**: Identify framework bottlenecks
3. **Run Nsight Compute**: Analyze specific kernels
4. **Enable Triton Profiler**: Optimize kernel generation
5. **Use HTA for Multi-GPU**: Analyze distributed performance
6. **Monitor Memory**: Track memory usage patterns

### 2. Modern GPU Specific
- **Multi-architecture Support**: Ensure proper targeting (sm_80, sm_86, sm_90)
- **Memory Bandwidth**: Monitor memory usage patterns
- **Tensor Cores**: Analyze matrix operation performance
- **Stream-ordered Memory**: Use `cudaMallocAsync`/`cudaFreeAsync`

### 3. PyTorch 2.8 Optimizations
- **torch.compile**: Enable with `mode="max-autotune"`
- **Dynamic Shapes**: Use `automatic_dynamic_shapes=True`
- **Memory Profiling**: Enable `profile_memory=True`
- **NVTX Integration**: Add custom markers

### 4. Triton 3.4 Features
- **Autotuning**: Use `triton.autotune_mode = "max-autotune"`
- **Kernel Fusion**: Enable kernel combination
- **Memory Coalescing**: Optimize memory access patterns
- **Occupancy Tuning**: Maximize GPU utilization

## Performance Metrics

### Key Metrics to Monitor
1. **GPU Utilization**: Target >90%
2. **Memory Bandwidth**: Varies by GPU model
3. **Tensor Core Usage**: For matrix operations
4. **Kernel Occupancy**: Maximize thread block efficiency
5. **Memory Latency**: Minimize access delays
6. **Communication Overhead**: For multi-GPU setups

### Expected Performance
- **CUDA 12.4**: 5-10% improvement over CUDA 12.0
- **PyTorch 2.8**: 15-25% improvement over PyTorch 2.7
- **Triton 3.4**: 10-15% improvement in kernel generation
- **Modern GPUs**: 20-30% improvement over older architectures

## Troubleshooting

### Common Issues
1. **Low GPU Utilization**: Check kernel occupancy and memory access
2. **High Memory Usage**: Enable memory profiling and optimization
3. **Poor Kernel Performance**: Use Nsight Compute for detailed analysis
4. **Multi-GPU Bottlenecks**: Use HTA for communication analysis

### Debugging Commands
```bash
# Check GPU status
nvidia-smi

# Monitor memory usage
watch -n 1 nvidia-smi

# Profile specific kernels
ncu --kernel-regex "gemm" your_script.py

# Analyze memory patterns
nsys profile --trace=cuda,cudamemcpy your_script.py
```

## Integration with Development

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
- name: Performance Testing
  run: |
    bash profiler_scripts/nsys_profile.sh tests/performance_test.py
    bash profiler_scripts/ncu_profile.sh tests/performance_test.py
```

### Automated Analysis
```python
# Example automated profiling script
import subprocess
import json

def run_profiling(script_path):
    # Run Nsight Systems
    subprocess.run([
        "nsys", "profile", "--force-overwrite=true",
        "-o", "profile_report", "python", script_path
    ])
    
    # Run Nsight Compute
    subprocess.run([
        "ncu", "--mode=launch", "--target-processes=python3",
        "--set", "full", "-o", "ncu_report", "python", script_path
    ])
```

## Conclusion

This profiling setup provides comprehensive analysis capabilities for the latest CUDA 12.4, PyTorch 2.8, and Triton 3.4 stack, with specific optimizations for modern GPUs. Use these tools in combination to achieve maximum performance and identify optimization opportunities.
