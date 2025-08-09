# Chapter 5: GPU-based Storage I/O Optimizations

## Summary
These examples demonstrate accelerating data pipelines with GPUDirect Storage, sequential-access patterns, pinned memory, and DataLoader tuning to raise end-to-end throughput.

## Performance Takeaways
- Use GPUDirect Storage to reduce CPU overhead and increase IO throughput
- Prefer sequential access patterns to achieve multi‑x read bandwidth gains
- Tune num_workers/prefetch/pin_memory for balanced, non‑starving pipelines
- Monitor I/O and GPU utilization to catch data bottlenecks early
- Achieve 2–10× vs random access; 20%+ throughput from GDS in favorable setups

This directory contains code examples that demonstrate the core storage I/O optimization concepts from Chapter 5 of the AI Performance Engineering book.

## Key Concepts Demonstrated

### 1. GPUDirect Storage (GDS)
- Direct GPU-to-storage transfers bypassing CPU
- Reduced CPU overhead for data movement
- Higher throughput for storage-bound workloads
- Lower latency for data transfers

### 2. Sequential vs Random Access Patterns
- Sequential access provides much better performance
- Random access patterns cause disk seek overhead
- Data layout optimization for sequential reads
- Impact on overall system throughput

### 3. Data Loading Optimization
- Optimal number of worker processes
- Pinned memory for faster GPU transfers
- Prefetching strategies
- Persistent workers to reduce overhead

### 4. Storage I/O Monitoring
- Real-time performance monitoring
- Bottleneck identification
- Resource utilization tracking
- Performance metrics collection

### 5. Memory Management
- Pinned memory allocation
- Prefetch factor optimization
- Memory usage monitoring
- Efficient data pipeline design

## Files

- `gpudirect_storage_example.py`: Main demonstration script showing GDS concepts, sequential vs random access, worker scaling, and memory optimization
- `storage_io_optimization.py`: Additional storage I/O optimization examples
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic GPUDirect Storage Example

```bash
python gpudirect_storage_example.py
```

This will run comprehensive demonstrations including:
- Sequential vs random access performance comparison
- Worker scaling analysis
- Memory optimization techniques
- GPUDirect Storage concepts
- Storage monitoring techniques

### Storage I/O Optimization

```bash
python storage_io_optimization.py
```

## Key Performance Insights

### 1. Sequential Access Benefits
- Sequential reads can be 2-10x faster than random access
- Large read sizes improve throughput
- Data layout optimization is crucial

### 2. Worker Process Scaling
- Optimal worker count depends on system resources
- Too few workers: underutilized CPU
- Too many workers: overhead and contention
- Monitor GPU utilization to find sweet spot

### 3. Memory Optimization
- Pinned memory improves GPU transfer performance
- Prefetch factor affects memory usage vs performance
- Balance between memory usage and throughput

### 4. GPUDirect Storage Benefits
- 20%+ throughput improvement in ideal scenarios
- Reduced CPU overhead
- Lower latency for data transfers
- Better resource utilization

## Monitoring and Profiling

### System Monitoring
```bash
# Monitor disk I/O
iostat -x 1

# Monitor GPU utilization
nvidia-smi -l 1

# Monitor system resources
htop
```

### PyTorch Profiling
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your data loading code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Best Practices

### 1. Data Layout
- Store data in large, sequential files
- Use appropriate file formats (Parquet, HDF5, etc.)
- Consider data compression for storage efficiency

### 2. Data Loading
- Use appropriate number of workers (typically 2-4x CPU cores)
- Enable pinned memory for GPU transfers
- Use persistent workers to reduce overhead
- Implement proper error handling

### 3. Monitoring
- Monitor disk I/O throughput
- Track GPU utilization during data loading
- Watch for memory pressure
- Profile data loading bottlenecks

### 4. Optimization
- Profile before optimizing
- Measure end-to-end performance
- Consider storage hardware capabilities
- Balance between throughput and latency

## Hardware Considerations

### Storage Types
- **NVMe SSDs**: High throughput, low latency
- **SATA SSDs**: Good performance, lower cost
- **HDDs**: High capacity, lower performance
- **Network Storage**: Shared access, network dependent

### GPU Memory
- Ensure sufficient GPU memory for data batches
- Consider unified memory for large datasets
- Monitor GPU memory usage patterns

### CPU and Memory
- Sufficient CPU cores for worker processes
- Adequate system memory for data caching
- Consider NUMA topology for multi-socket systems

## Troubleshooting

### Common Issues

1. **Low GPU Utilization During Data Loading**
   - Increase number of workers
   - Enable pinned memory
   - Check for I/O bottlenecks

2. **High Memory Usage**
   - Reduce prefetch factor
   - Use smaller batch sizes
   - Monitor memory leaks

3. **Poor Throughput**
   - Check disk I/O performance
   - Verify sequential access patterns
   - Profile data loading pipeline

4. **Worker Process Errors**
   - Check for memory issues
   - Verify data integrity
   - Implement proper error handling

## Advanced Topics

### 1. Multi-GPU Data Loading
- Distribute data loading across multiple GPUs
- Use NCCL for GPU-to-GPU communication
- Balance data distribution

### 2. Distributed Storage
- Network file systems (NFS, Lustre)
- Object storage integration
- Caching strategies

### 3. Custom Data Formats
- Optimized binary formats
- Compression techniques
- Streaming data formats

## References

- [GPUDirect Storage Documentation](https://developer.nvidia.com/gpudirect-storage)
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [NVIDIA Performance Tuning Guide](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html)
- [Storage Performance Best Practices](https://docs.nvidia.com/datacenter/tesla/pdfs/tesla-whitepaper.pdf)

## Performance Benchmarks

The examples include performance benchmarks that demonstrate:
- Sequential vs random access performance
- Worker scaling analysis
- Memory optimization impact
- GPUDirect Storage benefits

Run the examples on your system to see actual performance numbers and identify optimization opportunities.
