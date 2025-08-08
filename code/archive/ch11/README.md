# Chapter 11: Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations

This chapter covers advanced CUDA stream techniques for maximizing GPU utilization through inter-kernel concurrency.

## Code Examples

### Basic Streams (`basic_streams/`)
Demonstrates fundamental CUDA stream usage for overlapping kernel execution across multiple streams.

### Stream-Ordered Memory Allocator (`stream_ordered_allocator/`)
Shows how to use `cudaMallocAsync` and `cudaFreeAsync` for asynchronous memory management without global synchronization.

### Warp-Specialized Pipeline (`warp_specialized_pipeline/`)
Implements intra-kernel pipelining using warp specialization and the CUDA Pipeline API to hide memory latency.

### Multi-Stream Pipeline (`multi_stream_pipeline/`)
Combines warp-specialized kernels with multiple CUDA streams for two-layer pipelining (intra-kernel + inter-kernel).

### Programmatic Dependent Launch (`pdl_example/`)
Demonstrates PDL for inter-kernel overlap using `cudaTriggerProgrammaticLaunchCompletion()` and `cudaGridDependencySynchronize()`.

### Combined PDL + Thread Block Clusters (`combined_pdl_cluster/`)
Shows the pinnacle of CUDA optimizations by combining PDL, thread block clusters, and warp specialization.

## Key Concepts

- **CUDA Streams**: Independent operation queues for overlapping work
- **Stream-Ordered Memory**: Asynchronous allocation/deallocation without global synchronization
- **Warp Specialization**: Dividing work across different warps within a thread block
- **Programmatic Dependent Launch**: Device-side kernel triggering for fine-grained overlap
- **Thread Block Clusters**: Multi-SM cooperation for large-scale workloads

## Building and Running

Each example can be built and run independently:

```bash
cd basic_streams
make
make run
```

## Dependencies

- CUDA 12.9+
- NVIDIA GPU with compute capability 9.0+
- C++17 support for advanced features

## Performance Considerations

- Use explicit streams instead of the default stream
- Enable stream-ordered memory allocation for LLM workloads
- Combine intra-kernel and inter-kernel pipelining for maximum utilization
- Consider complexity vs. performance trade-offs for your specific workload
