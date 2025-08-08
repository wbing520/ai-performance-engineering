# Performance Profiling Guide

## Overview
This guide covers the latest profiling tools and best practices for:
- **CUDA 12.9**
- **PyTorch 2.8 nightly**
- **OpenAI Triton 3.4**
- **Architecture switching: Hopper H100/H200 and Blackwell B200/B300 GPUs**

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

**Key Features for CUDA 12.9**:
- Python backtrace sampling
- CUDA backtrace integration
- Multi-GPU support
- Hardware metrics collection
- Blackwell B200/B300 specific metrics

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

**Key Features for Architecture Support**:
- SM90 (Hopper H100/H200) and SM100 (Blackwell B200/B300) architecture support
- Tensor Core metrics
- HBM3/HBM3e memory analysis
- Advanced occupancy analysis
- TMA (Tensor Memory Accelerator) metrics for Blackwell
- Transformer Engine metrics for Hopper

### 3. PyTorch Profiler
**Latest Version**: PyTorch 2.8 nightly
**Purpose**: Framework-level profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx

# Latest PyTorch 2.8 nightly profiler configuration
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

**Key Features for PyTorch 2.8 nightly**:
- Enhanced memory profiling
- FLOP counting
- Module-level analysis
- TensorBoard integration
- Architecture-specific optimizations (Hopper and Blackwell)

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
- Architecture-specific optimizations (Hopper and Blackwell)

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
- NVLink-C2C analysis

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
- HBM3e memory analysis

### 7. Perf Profiler
**Purpose**: System-level performance analysis

```bash
# Perf profiling
perf record -g -p $(pgrep python) -o perf.data
perf report -i perf.data
```

**Key Features**:
- CPU performance analysis
- System call analysis
- Hardware event monitoring
- Call graph analysis

## Best Practices

### 1. Profiling Workflow
1. **Start with Nsight Systems**: Get system-level overview
2. **Use PyTorch Profiler**: Identify framework bottlenecks
3. **Run Nsight Compute**: Analyze specific kernels
4. **Enable Triton Profiler**: Optimize kernel generation
5. **Use HTA for Multi-GPU**: Analyze distributed performance
6. **Monitor Memory**: Track memory usage patterns
7. **Use Perf**: System-level analysis

### 2. Architecture-Specific Considerations
- **Hopper H100/H200 (SM90)**: 
  - **HBM3 Memory**: Monitor high-bandwidth memory usage
  - **Transformer Engine**: Analyze transformer-specific optimizations
  - **Dynamic Programming**: Monitor dynamic programming features
- **Blackwell B200/B300 (SM100)**:
  - **HBM3e Memory**: Monitor high-bandwidth memory usage
  - **TMA**: Tensor Memory Accelerator analysis
  - **Stream-ordered Memory**: Use `cudaMallocAsync`/`cudaFreeAsync`
  - **NVLink-C2C**: Direct GPU-to-GPU communication

### 3. PyTorch 2.8 Nightly Optimizations
- **torch.compile**: Enable with `mode="max-autotune"`
- **Dynamic Shapes**: Use `automatic_dynamic_shapes=True`
- **Memory Profiling**: Enable `profile_memory=True`
- **NVTX Integration**: Add custom markers
- **Architecture Optimizations**: Enable architecture-specific features

### 4. Triton 3.4 Features
- **Autotuning**: Use `triton.autotune_mode = "max-autotune"`
- **Kernel Fusion**: Enable kernel combination
- **Memory Coalescing**: Optimize memory access patterns
- **Occupancy Tuning**: Maximize GPU utilization
- **Architecture Support**: Leverage architecture-specific features

## Performance Metrics

### Key Metrics to Monitor
1. **GPU Utilization**: Target >90%
2. **Memory Bandwidth**: HBM3e @ 3.2 TB/s
3. **Tensor Core Usage**: For matrix operations
4. **Kernel Occupancy**: Maximize thread block efficiency
5. **Memory Latency**: Minimize access delays
6. **Communication Overhead**: For multi-GPU setups
7. **TMA Efficiency**: Tensor Memory Accelerator usage

### Expected Performance
- **CUDA 12.9**: 10-15% improvement over CUDA 12.4
- **Hopper H200**: 20-30% improvement over H100
- **Blackwell B200/B300**: 30-50% improvement over H100
- **PyTorch 2.8 nightly**: 20-30% improvement over stable releases
- **Triton 3.4**: 15-20% improvement in kernel generation

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

# System-level analysis
perf record -g -p $(pgrep python)
```

## Integration with Development

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
- name: Performance Testing
  run: |
    bash profiler_scripts/nsys_profile.sh tests/performance_test.py
    bash profiler_scripts/ncu_profile.sh tests/performance_test.py
    bash profiler_scripts/hta_profile.sh tests/performance_test.py
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
    
    # Run HTA
    subprocess.run([
        "nsys", "profile", "--force-overwrite=true",
        "-o", "hta_report", "-t", "cuda,nvtx,nccl",
        "python", script_path
    ])
```

## Conclusion

This profiling setup provides comprehensive analysis capabilities for the latest CUDA 12.9, PyTorch 2.8 nightly, and Triton 3.4 stack, with specific optimizations for Blackwell B200/B300 GPUs. Use these tools in combination to achieve maximum performance and identify optimization opportunities.
