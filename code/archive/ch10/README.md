# Chapter 10: Advanced CUDA Features

This chapter demonstrates advanced CUDA programming features including **CUDA Graphs**, **Dynamic Parallelism**, **Unified Memory**, **Stream-ordered Memory Allocation**, and **Advanced Synchronization**. These features are essential for achieving maximum performance on modern GPUs.

## Running on 8× B200 Cluster with Grace CPU

### CUDA Graphs

```bash
cd code/ch10

# Compile CUDA Graph examples
nvcc -arch=sm_100 -o cuda_graphs cuda_graphs.cu
nvcc -arch=sm_100 -o graph_capture graph_capture.cu

# Run CUDA Graph examples
./cuda_graphs
./graph_capture

# Run PyTorch CUDA Graph example
python pytorch_graph.py
```

### Dynamic Parallelism

```bash
# Compile dynamic parallelism examples
nvcc -arch=sm_100 -o dynamic_parallelism dynamic_parallelism.cu
nvcc -arch=sm_100 -o nested_kernels nested_kernels.cu

# Run dynamic parallelism examples
./dynamic_parallelism
./nested_kernels
```

### Unified Memory

```bash
# Compile unified memory examples
nvcc -arch=sm_100 -o unified_memory unified_memory.cu
nvcc -arch=sm_100 -o managed_memory managed_memory.cu

# Run unified memory examples
./unified_memory
./managed_memory

# Run PyTorch unified memory example
python pytorch_unified_memory.py
```

### Stream-ordered Memory

```bash
# Compile stream-ordered memory examples
nvcc -arch=sm_100 -o stream_ordered_memory stream_ordered_memory.cu
nvcc -arch=sm_100 -o async_memory async_memory.cu

# Run stream-ordered memory examples
./stream_ordered_memory
./async_memory
```

### Advanced Synchronization

```bash
# Compile synchronization examples
nvcc -arch=sm_100 -o events_synchronization events_synchronization.cu
nvcc -arch=sm_100 -o cooperative_groups cooperative_groups.cu

# Run synchronization examples
./events_synchronization
./cooperative_groups
```

## Profiling Commands

### Nsight Systems (nsys)

```bash
# Profile CUDA Graph execution
nsys profile -t cuda,osrt -o cuda_graphs_profile ./cuda_graphs
nsys profile -t cuda,nvtx -o graph_capture_profile ./graph_capture

# Profile dynamic parallelism
nsys profile -t cuda,osrt -o dynamic_parallelism_profile ./dynamic_parallelism
nsys profile -t cuda,osrt -o nested_kernels_profile ./nested_kernels

# Profile unified memory
nsys profile -t cuda,osrt -o unified_memory_profile ./unified_memory
nsys profile -t cuda,osrt -o managed_memory_profile ./managed_memory

# Profile stream-ordered memory
nsys profile -t cuda,osrt -o stream_ordered_memory_profile ./stream_ordered_memory
nsys profile -t cuda,osrt -o async_memory_profile ./async_memory

# Profile synchronization
nsys profile -t cuda,osrt -o events_sync_profile ./events_synchronization
nsys profile -t cuda,osrt -o cooperative_groups_profile ./cooperative_groups
```

### Nsight Compute (ncu)

```bash
# Profile kernel efficiency for advanced features
ncu --metrics achieved_occupancy,warp_execution_efficiency -o graph_kernel_profile ./cuda_graphs
ncu --metrics achieved_occupancy,warp_execution_efficiency -o dynamic_kernel_profile ./dynamic_parallelism
ncu --metrics achieved_occupancy,warp_execution_efficiency -o unified_kernel_profile ./unified_memory
ncu --metrics achieved_occupancy,warp_execution_efficiency -o stream_kernel_profile ./stream_ordered_memory

# Profile memory throughput
ncu --metrics dram_read_throughput,dram_write_throughput -o memory_profile ./unified_memory
ncu --metrics dram_read_throughput,dram_write_throughput -o stream_memory_profile ./stream_ordered_memory
```

### PyTorch Profiler

```bash
# Profile PyTorch CUDA Graph usage
python -m torch.utils.bottleneck pytorch_graph.py

# Profile PyTorch unified memory
python -m torch.utils.bottleneck pytorch_unified_memory.py

# Profile with memory tracking
python -c "
import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    import pytorch_graph
    pytorch_graph.main()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
"
```

### Memory Profiling

```bash
# Monitor unified memory usage
python -c "
import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

print(f'Initial memory: {torch.cuda.memory_allocated()/1e6:.2f} MB')
import pytorch_unified_memory
pytorch_unified_memory.main()
print(f'Final memory: {torch.cuda.memory_allocated()/1e6:.2f} MB')
"
```

## Expected Output

### CUDA Graphs

```
# cuda_graphs.cu
CUDA Graph created successfully
Graph execution time: 0.5 ms
Regular execution time: 0.8 ms
Speedup: 1.6x

# graph_capture.cu
Graph captured with 3 operations
Graph execution completed
Memory usage optimized
```

### Dynamic Parallelism

```
# dynamic_parallelism.cu
Parent kernel launched
Child kernel launched from device
Dynamic parallelism completed
Total threads: 1024 + 512 = 1536

# nested_kernels.cu
Nested kernel execution completed
Kernel depth: 3 levels
Total operations: 1000
```

### Unified Memory

```
# unified_memory.cu
Unified memory allocation: 100 MB
CPU-GPU transfer time: 0.2 ms
GPU computation time: 1.5 ms
Total time: 1.7 ms

# managed_memory.cu
Managed memory allocation successful
Page fault handling optimized
Memory migration: CPU -> GPU -> CPU
```

### Stream-ordered Memory

```
# stream_ordered_memory.cu
Stream-ordered allocation: 50 MB
Allocation time: 0.1 ms
Deallocation time: 0.05 ms
Memory fragmentation: 0%

# async_memory.cu
Async memory operations completed
Stream synchronization optimized
Memory pool utilization: 95%
```

### Advanced Synchronization

```
# events_synchronization.cu
CUDA events created
Event synchronization: 0.1 ms
Stream synchronization: 0.3 ms
Timing accuracy: 1 microsecond

# cooperative_groups.cu
Cooperative groups initialized
Grid-level synchronization completed
Block-level cooperation: 32 threads
Warp-level cooperation: 32 threads
```

### PyTorch Examples

```
# pytorch_graph.py
PyTorch CUDA Graph captured
Graph execution time: 2.1 ms
Regular execution time: 3.5 ms
Memory usage: 512 MB

# pytorch_unified_memory.py
Unified memory tensor created
CPU-GPU access time: 0.5 ms
Memory migration: automatic
Page fault handling: optimized
```

## Tuning Tips

1. **CUDA Graphs**: Capture once, execute many times for reduced overhead
2. **Dynamic Parallelism**: Use for irregular workloads and nested parallelism
3. **Unified Memory**: Enable automatic memory management for complex data structures
4. **Stream-ordered Memory**: Reduce fragmentation and improve allocation performance
5. **Advanced Synchronization**: Use appropriate synchronization primitives for your use case

## Troubleshooting

- **Graph Capture Failed**: Ensure all operations are graph-compatible
- **Dynamic Parallelism**: Check compute capability support (SM 3.5+)
- **Unified Memory**: Monitor page fault overhead
- **Stream-ordered Memory**: Verify CUDA version support (11.2+)
- **Synchronization**: Use proper event ordering and stream management

## Architecture-Specific Notes

### Blackwell B200/B300 with Grace CPU

- **CUDA Graphs**: Optimized for HBM3e memory bandwidth
- **Dynamic Parallelism**: Full support with SM100
- **Unified Memory**: HBM3e provides faster CPU-GPU access
- **Stream-ordered Memory**: Optimized for Blackwell architecture
- **Advanced Synchronization**: Hardware-accelerated events and barriers

### CUDA 12.9 Optimizations

- **Graph Replay**: Optimized for Blackwell SM100
- **Memory Management**: Enhanced unified memory with HBM3e
- **Stream Operations**: Improved stream-ordered allocation
- **Synchronization**: Hardware-accelerated cooperative groups

## Performance Analysis

### CUDA Graphs Benefits

- **Reduced Launch Overhead**: ~60% reduction in kernel launch time
- **Memory Optimization**: Better memory allocation patterns
- **Predictable Performance**: Consistent execution times
- **Multi-GPU**: Efficient graph distribution across GPUs

### Dynamic Parallelism Advantages

- **Irregular Workloads**: Adaptive kernel launching
- **Nested Parallelism**: Hierarchical computation patterns
- **Load Balancing**: Dynamic work distribution
- **Memory Efficiency**: On-demand memory allocation

### Unified Memory Performance

- **Automatic Management**: Simplified memory handling
- **Page Migration**: Optimized CPU-GPU data movement
- **HBM3e Benefits**: Faster unified memory access
- **Multi-GPU**: Seamless memory sharing

This chapter provides the advanced CUDA features needed for maximum performance on modern GPUs, with specific optimizations for the Blackwell B200/B300 architecture.
