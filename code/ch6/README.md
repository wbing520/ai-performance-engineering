# Chapter 6: GPU Architecture, CUDA Programming, and Maximizing Occupancy

## Summary
These examples demonstrate core CUDA programming patterns and occupancy tuning (launch parameters, unified and stream-ordered memory) to maximize parallel performance.

## Performance Takeaways
- Choose launch configurations that raise occupancy without overusing registers
- Apply stream‑ordered allocation to avoid global syncs and allocator stalls
- Use unified memory prefetch/advice to match access patterns and reduce faults
- Translate occupancy and memory metrics into end‑to‑end kernel speedups
- Leverage the Occupancy API to compute optimal block sizes per kernel

This chapter introduces GPU architecture fundamentals and CUDA programming patterns for AI workloads, focusing on memory hierarchy, thread execution models, and occupancy optimization.

## Code Examples

### Basic CUDA Programming
- `my_first_kernel.cu` - First CUDA kernel example with memory management
- `simple_kernel.cu` - Improved kernel with dynamic launch parameters
- `2d_kernel.cu` - 2D kernel for image/matrix processing

### Performance Comparison
- `add_sequential.cu` / `add_sequential.py` - Sequential vector addition (poor performance)
- `add_parallel.cu` / `add_parallel.py` - Parallel vector addition (optimal performance)

### Advanced Memory Management
- `stream_ordered_allocator.cu` - Stream-ordered memory allocation with cudaMallocAsync
- `unified_memory.cu` - CUDA Managed Memory with prefetching and advice

### Occupancy Optimization
- `launch_bounds_example.cu` - Using __launch_bounds__ for occupancy tuning
- `occupancy_api.cu` - CUDA Occupancy API for optimal block size selection

## Key Concepts

1. **GPU Architecture**: SIMT execution model, warps, thread blocks, and grids
2. **Memory Hierarchy**: Registers → L1/Shared → L2 → Global (HBM3e) → Host
3. **Thread Organization**: Threads (32 per warp) → Blocks (up to 1024 threads) → Grids
4. **Occupancy**: Ratio of active warps to maximum possible warps per SM
5. **Launch Parameters**: Choosing optimal threadsPerBlock and blocksPerGrid
6. **Asynchronous Operations**: Stream-ordered allocation and unified memory

## Hardware Limits (Blackwell B200)

- **Warp size**: 32 threads (fundamental SIMT unit)
- **Max threads per block**: 1,024 threads
- **Max warps per block**: 32 warps
- **Max resident warps per SM**: 64 warps (2,048 threads)
- **Max active blocks per SM**: 32 blocks
- **Shared memory per SM**: 228 KB (user-allocatable)
- **Register file per SM**: 256 KB (64K 32-bit registers)

## Requirements

- CUDA 12.8+
- NVIDIA GPU with compute capability 5.0+
- For optimal performance: Blackwell B200/B300 (SM100)

## Building and Running

```bash
# Compile CUDA examples
nvcc -o my_first_kernel my_first_kernel.cu
nvcc -o add_parallel add_parallel.cu
nvcc -o occupancy_api occupancy_api.cu

# Run examples
./my_first_kernel
./add_parallel
python add_parallel.py

# Profile with Nsight Systems
nsys profile -o add_parallel ./add_parallel
nsys stats --report summary,cuda_api,osrt --format sqlite,csv add_parallel -o add_parallel

# Profile with Nsight Compute
ncu --metrics achieved_occupancy,warp_execution_efficiency ./occupancy_api
```

## Profiling Commands

```bash
# Compare sequential vs parallel
nsys profile -o sequential ./add_sequential
nsys stats --report summary,cuda_api,osrt --format sqlite,csv sequential -o sequential
nsys profile -o parallel ./add_parallel
nsys stats --report summary,cuda_api,osrt --format sqlite,csv parallel -o parallel

# Analyze occupancy
ncu --metrics achieved_occupancy,sm__cycles_active.avg.pct_of_peak_sustained_elapsed ./occupancy_api

# Memory analysis
ncu --section MemoryWorkloadAnalysis ./unified_memory
```