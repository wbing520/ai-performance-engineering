# Chapter 8: Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism

## Summary
These examples demonstrate keeping SMs busy through occupancy tuning, reducing warp divergence, and exploiting instruction‑level parallelism, with profiling to validate gains.

## Performance Takeaways
- Reduce warp divergence via predication and structured control flow
- Increase ILP with loop unrolling and independent operations
- Balance occupancy with register/shared‑memory pressure for net gains
- Validate improvements with warp efficiency and occupancy metrics
- Translate kernel‑level wins into application throughput improvements

This chapter covers advanced GPU optimization techniques focusing on keeping the GPU fully utilized through better warp scheduling and parallelism.

## Examples

### Occupancy Tuning
- `occupancy_tuning.cu` - Demonstrates using __launch_bounds__ and occupancy API
- `occupancy_api_example.cu` - Using CUDA occupancy API to find optimal block sizes
- `occupancy_pytorch.py` - PyTorch considerations for occupancy

### Warp Divergence and Efficiency
- `threshold_naive.cu` - Example with warp divergence issues
- `threshold_predicated.cu` - Optimized version using predication
- `warp_divergence_pytorch.py` - PyTorch approaches to avoid divergence

### Instruction-Level Parallelism (ILP)
- `ilp_basic.cu` - Basic instruction-level parallelism example
- `loop_unrolling.cu` - Loop unrolling for better ILP
- `independent_ops.cu` - Separating independent operations
- `ilp_pytorch.py` - PyTorch approaches to ILP

### Profiling and Analysis
- `profiling_example.cu` - Kernel designed for profiling analysis
- `profile_analysis.py` - PyTorch profiling with torch.profiler

## Key Concepts

1. **Occupancy**: Ratio of active warps to maximum possible warps per SM
2. **Warp Divergence**: When threads in a warp take different execution paths
3. **Instruction-Level Parallelism**: Multiple independent operations per thread
4. **Latency Hiding**: Using parallelism to hide memory and compute latencies

## Profiling Commands

```bash
# Nsight Compute for detailed kernel analysis
ncu --metrics smsp__warps_active.avg.pct_of_peak_sustained,smsp__warp_execution_efficiency.avg ./kernel

# Nsight Systems for timeline analysis
nsys profile --trace=cuda,nvtx ./application

# PyTorch profiler
python -c "import torch; torch.profiler.profile(...)"
```

## Requirements

- CUDA 12.8+
- PyTorch 2.8+
- Nsight Compute/Systems for profiling
