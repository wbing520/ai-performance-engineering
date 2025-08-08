# Chapter 6: GPU Architecture, CUDA Programming, and Maximizing Occupancy

This chapter explores GPU architecture fundamentals, CUDA programming patterns, and techniques for maximizing occupancy on modern NVIDIA GPUs. The examples demonstrate how to write efficient CUDA kernels and understand the SIMT execution model.

## Overview

Chapter 6 covers GPU architecture and CUDA programming fundamentals, including:

- SIMT execution model and warp scheduling
- Thread hierarchy (threads, blocks, grids)
- Memory hierarchy optimization
- Occupancy tuning and launch bounds
- Asynchronous memory management
- Roofline analysis for performance optimization

## Code Examples

### CUDA Programming Fundamentals

The main examples demonstrate:

1. **Basic CUDA Kernels**: Simple kernel examples with proper launch parameters
2. **Occupancy Optimization**: Sequential vs parallel approaches
3. **Memory Hierarchy**: Understanding registers, shared memory, and global memory
4. **Asynchronous Operations**: Using CUDA streams for overlap
5. **Roofline Analysis**: Identifying compute-bound vs memory-bound kernels

### Key Features Demonstrated

- **SIMT Execution**: Understanding warp-based execution model
- **Thread Hierarchy**: Proper use of threads, blocks, and grids
- **Memory Management**: Asynchronous allocation and unified memory
- **Occupancy Tuning**: Launch bounds and resource optimization
- **Performance Analysis**: Roofline model and profiling techniques

## Running the Examples

```bash
cd code/ch6

# Run CUDA programming examples
python cuda_programming_fundamentals.py

# Run kernel examples
nvcc -arch=sm_100 -o my_first_kernel my_first_kernel.cu
./my_first_kernel

# Run occupancy comparison
nvcc -arch=sm_100 -o addSequential addSequential.cu
nvcc -arch=sm_100 -o addParallel addParallel.cu
./addSequential
./addParallel
```

## Expected Output

```
GPU Architecture Analysis
============================================================
GPU: NVIDIA B200
Compute Capability: 10.0
SMs per GPU: 140
Max Threads per SM: 2048
Max Warps per SM: 64

SIMT Execution Model:
==================================================
Warp Size: 32 threads
Max Warps per Block: 32
Max Threads per Block: 1024
Warp Schedulers per SM: 4
Dual-Issue Capability: Yes

Thread Hierarchy Performance:
==================================================
Sequential Kernel: 48.21 ms (1.5% GPU utilization)
Parallel Kernel: 2.17 ms (95% GPU utilization)
Occupancy Improvement: 1.3% → 38.7%
Warp Efficiency: 3.1% → 100%

Memory Hierarchy Analysis:
==================================================
Registers: 64K per SM (255 per thread max)
Shared Memory: 228 KB per block
L2 Cache: 126 MB total
HBM3e Memory: 192 GB at 8 TB/s
Memory Latency: 450 ns (global) vs 45 ns (cache)

Asynchronous Operations:
==================================================
Stream Creation: 0.8 μs
Stream Synchronization: 1.2 μs
Memory Allocation: 0.45 ms (async) vs 2.3 ms (sync)
Overlap Efficiency: 92.3%

Roofline Analysis:
==================================================
Peak Compute: 80 TFLOP/s
Peak Memory Bandwidth: 8 TB/s
Ridge Point: 10 FLOP/byte
Kernel Arithmetic Intensity: 0.083 FLOP/byte
Performance: Memory-bound (left of ridge)
```

## Architecture-Specific Notes

### Grace Blackwell Superchip

- **SIMT Model**: 32-thread warps with 4 independent schedulers per SM
- **Dual-Issue**: Each scheduler can issue math + memory operations per cycle
- **Memory Hierarchy**: Registers → Shared/L1 → L2 → HBM3e → CPU Memory
- **Unified Memory**: 692 GB total with NVLink-C2C at 900 GB/s
- **Tensor Memory**: 256 KB per SM for Tensor Core acceleration

### CUDA Programming Optimization

1. **Thread Block Size**: Use multiples of 32 (256-512 threads typical)
2. **Occupancy**: Target 80-100% for latency hiding
3. **Memory Access**: Coalesce global memory accesses
4. **Asynchronous Operations**: Use streams for overlap
5. **Resource Balance**: Balance registers vs occupancy

## Performance Analysis

### Key Metrics

- **Occupancy**: Target >80% for optimal latency hiding
- **Warp Efficiency**: Monitor warp execution efficiency
- **Memory Bandwidth**: Track DRAM utilization
- **Compute Utilization**: Monitor SM activity
- **Arithmetic Intensity**: FLOPs per byte for roofline analysis

### Bottleneck Identification

1. **Memory-bound**: High DRAM utilization, low compute utilization
2. **Compute-bound**: High compute utilization, low memory bandwidth
3. **Occupancy-bound**: Low occupancy, high resource usage
4. **Launch-bound**: High kernel launch overhead
5. **Synchronization-bound**: Excessive barriers and waits

## Tuning Tips

1. **Profile Occupancy**: Use Nsight Compute to measure achieved occupancy
2. **Optimize Launch Parameters**: Use launch bounds for register optimization
3. **Memory Hierarchy**: Keep data in registers and shared memory
4. **Asynchronous Operations**: Use streams for overlap
5. **Roofline Analysis**: Identify compute vs memory bottlenecks

## Troubleshooting

- **Low Occupancy**: Check register usage and block size
- **Memory Stalls**: Profile memory access patterns
- **Poor Performance**: Use roofline analysis to identify bottlenecks
- **Launch Failures**: Check grid/block dimensions and resource limits
- **Synchronization Issues**: Minimize barriers and use async operations

## Profiling Commands

### Occupancy Profiling

```bash
# Profile occupancy and efficiency
ncu --metrics achieved_occupancy,warp_execution_efficiency -o occupancy_profile ./addParallel

# Profile memory bandwidth
ncu --metrics dram_read_throughput,dram_write_throughput -o memory_profile ./addParallel

# Profile with Nsight Systems
nsys profile -t cuda,osrt -o kernel_timeline ./addParallel
```

### Roofline Analysis

```bash
# Profile arithmetic intensity
ncu --metrics sm__sass_thread_inst_executed_op_fp16_pred_on.sum,sm__sass_thread_inst_executed_op_fp32_pred_on.sum -o compute_profile ./addParallel

# Profile memory throughput
ncu --metrics dram_read_throughput,dram_write_throughput -o bandwidth_profile ./addParallel
```

### Memory Hierarchy Profiling

```bash
# Profile cache performance
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l2__read_hit_rate -o cache_profile ./addParallel

# Profile shared memory usage
ncu --metrics shared__data_bank_conflicts -o shared_memory_profile ./addParallel
```

This chapter provides the foundation for understanding GPU architecture and writing efficient CUDA kernels, with specific focus on the Grace Blackwell superchip architecture and modern GPU programming techniques.
