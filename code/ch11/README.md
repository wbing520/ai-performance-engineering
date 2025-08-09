# Chapter 11: Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations

## Summary
These examples demonstrate inter‑kernel concurrency with CUDA streams, fine‑grained synchronization, and stream‑ordered memory allocation for true compute–copy overlap.

## Performance Takeaways
- Structure multi‑stream pipelines to keep copy and compute engines busy
- Replace blocking synchronizations with events to preserve overlap
- Enable stream‑ordered allocation to eliminate global allocator barriers
- Reliably overlap H2D/compute/D2H for higher sustained throughput
- Realize 2–3× throughput gains in concurrent pipelines on modern GPUs

This chapter focuses on inter-kernel concurrency using CUDA streams, fine-grained synchronization, and stream-ordered memory allocation to achieve maximum GPU utilization.

## Code Examples

### CUDA Streams
- `basic_streams.cu` - Basic CUDA streams with kernel and copy overlap
- `stream_ordered_allocator.cu` - Stream-ordered memory allocation examples
- `multi_stream_pipeline.cu` - Combined intra-kernel and inter-kernel pipelining

## Key Concepts

### CUDA Streams
A CUDA stream is a sequence of operations that execute in order on the GPU while allowing multiple streams to run concurrently.

**Benefits:**
- Overlap kernel execution across streams
- Overlap compute with data transfers
- Hide memory allocation/deallocation overhead
- Enable fine-grained synchronization

### Stream-Ordered Memory Allocation
Modern CUDA provides `cudaMallocAsync()` and `cudaFreeAsync()` for stream-ordered memory operations.

```cuda
// Stream-ordered allocation (non-blocking)
cudaMallocAsync(&ptr, size, stream);
cudaFreeAsync(ptr, stream);

// vs Traditional allocation (blocking)
cudaMalloc(&ptr, size);  // Blocks all streams!
cudaFree(ptr);           // Blocks all streams!
```

**Key Advantages:**
- No global device synchronization
- Perfect for variable-length sequences (LLMs)
- Enables true overlap of memory management and compute
- Memory pool reduces OS allocation overhead

### Default Stream Pitfalls
- **Legacy Default Stream (Stream 0)**: Blocks and is blocked by all other streams
- **Per-Thread Default Streams (PTDS)**: Each CPU thread gets independent default stream
- **Best Practice**: Use explicit streams for performance-critical code

```bash
# Enable PTDS
export CUDA_API_PER_THREAD_DEFAULT_STREAM=1
```

### Fine-Grained Synchronization
Use CUDA events for precise inter-stream coordination without blocking the CPU.

```cuda
// Producer stream
cudaEventRecord(data_ready, producer_stream);

// Consumer stream  
cudaStreamWaitEvent(consumer_stream, data_ready, 0);
// Now consumer can proceed knowing data is ready
```

## Performance Patterns

### Three-Way Overlap
Modern GPUs support simultaneous:
1. **Host→Device (H2D) transfers** - Dedicated DMA engine
2. **Kernel computation** - SM compute units
3. **Device→Host (D2H) transfers** - Dedicated DMA engine

### LLM Workload Optimization
Stream-ordered allocation is crucial for:
- Variable-length sequences with different memory requirements
- Dynamic scratch buffer allocation per batch
- Overlapping attention computation with data movement
- Multi-layer pipelining across GPU boundaries

### Warp Specialization + Streams
Combine intra-kernel and inter-kernel pipelining:
- **Intra-kernel**: Warp specialization within each kernel
- **Inter-kernel**: Multiple streams processing different batches
- **Result**: Maximum hardware utilization across all GPU engines

## Building and Running

### Compilation
```bash
make all                    # Build all examples
make basic_streams          # Build basic streams example
make stream_ordered_allocator  # Build allocation example
make multi_stream_pipeline  # Build combined pipeline
```

### Execution
```bash
# Basic streams demonstration
./basic_streams

# Stream-ordered allocation comparison
./stream_ordered_allocator

# Multi-stream pipeline with warp specialization
./multi_stream_pipeline
```

## Performance Analysis

### Key Metrics to Monitor
- **Stream utilization**: Multiple streams active simultaneously
- **Memory bandwidth**: Overlap of H2D, compute, and D2H
- **Allocation overhead**: Stream-ordered vs traditional allocation
- **Synchronization efficiency**: Event-based vs blocking synchronization

### Expected Performance Gains
- **2-3x throughput** with proper stream overlap
- **50-80% reduction** in allocation overhead with stream-ordered allocator
- **Near-linear scaling** with number of concurrent streams (up to hardware limits)

### Hardware Limits
- **Concurrent kernels**: ~128 on modern GPUs
- **Copy engines**: 2 (H2D and D2H) on most GPUs
- **Memory bandwidth**: ~1-2 TB/s on high-end GPUs
- **SM resources**: Shared among concurrent kernels

## Profiling and Debugging

### Nsight Systems Analysis
```bash
# Stream timeline visualization
nsys profile --force-overwrite=true -o streams_timeline ./basic_streams

# Memory allocation analysis
nsys profile --force-overwrite=true -o alloc_analysis ./stream_ordered_allocator

# Combined pipeline analysis
nsys profile --force-overwrite=true -o pipeline_analysis ./multi_stream_pipeline
```

### Nsight Compute Analysis
```bash
# Launch statistics and stream utilization
ncu --section LaunchStats --section MemoryWorkloadAnalysis ./basic_streams

# Memory allocation patterns
ncu --section MemoryWorkloadAnalysis ./stream_ordered_allocator

# Warp efficiency in multi-stream context
ncu --section WarpStateStats ./multi_stream_pipeline
```

### Key Timeline Patterns to Look For
- **Overlapping operations**: Kernels and copies running simultaneously
- **No gaps**: Continuous GPU utilization without idle periods
- **Proper synchronization**: Events coordinating streams without over-synchronization
- **Memory allocation**: Non-blocking allocation patterns

## PyTorch Integration

### Enable Stream-Ordered Allocation
```bash
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
```

This automatically enables stream-ordered allocation in PyTorch, providing:
- Faster memory allocation/deallocation
- Better overlap with computation
- Reduced global synchronization

### DataLoader Optimization
```python
# Enable pinned memory for async transfers
dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)

# PyTorch automatically uses non-default streams for:
# - cuDNN operations
# - cuBLAS operations  
# - NCCL communications
```

## Best Practices

### Stream Management
1. **Create explicit streams** for all performance-critical operations
2. **Avoid default stream (Stream 0)** - it creates global barriers
3. **Enable PTDS** when using multiple CPU threads
4. **Use stream-ordered allocation** for variable-length workloads
5. **Balance stream count** - too many streams can reduce per-stream resources

### Synchronization
1. **Use events** instead of `cudaStreamSynchronize()` when possible
2. **Minimize global barriers** - they destroy overlap
3. **Event granularity** - don't over-synchronize, but ensure correctness
4. **Host callbacks** for CPU-GPU coordination without polling

### Memory Management
1. **Pinned host memory** required for true async transfers
2. **Stream-ordered allocation** for dynamic memory needs
3. **Memory pool tuning** - adjust release threshold based on workload
4. **Batch sizes** - balance memory usage with parallelism

### Debugging Guidelines
1. **Timeline analysis** - use Nsight Systems to visualize stream overlap
2. **Start simple** - verify single stream before adding complexity
3. **Check dependencies** - ensure proper event-based synchronization
4. **Resource limits** - monitor concurrent kernel limits and memory usage

## Advanced Topics

### Thread Block Clusters + Streams
Combining the most advanced GPU features:
- Thread block clusters for grid-level cooperation
- Warp specialization within clusters
- Multiple streams for inter-kernel pipelining

**Complexity Warning**: Extremely complex to debug and tune. Most workloads achieve 90%+ of peak performance with simpler approaches.

### Megakernels and Persistent Kernels
For ultra-low latency inference:
- Single large kernel processing entire model
- Eliminates kernel launch overhead
- Requires careful resource management
- Used in specialized inference engines

## Requirements

- CUDA 11.0+ (CUDA 12.8+ recommended for all features)
- Compute Capability 6.0+ (7.0+ recommended)
- Memory pools support (most modern GPUs)
- Multiple SMs for effective concurrency

## Troubleshooting

### Common Issues
1. **No overlap observed**: Check for default stream usage or blocking operations
2. **Poor performance**: Verify pinned memory usage for async transfers
3. **Allocation slowdowns**: Ensure stream-ordered allocation is enabled
4. **Deadlocks**: Check event dependencies and avoid circular waits

### Performance Debugging
1. **Profile first**: Use Nsight Systems to identify bottlenecks
2. **Check hardware limits**: Verify not hitting concurrent kernel limits
3. **Memory bandwidth**: Ensure not saturating memory subsystem
4. **Stream balance**: Adjust number of streams based on workload characteristics