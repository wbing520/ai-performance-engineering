# Chapter 7: Profiling and Tuning GPU Memory Access Patterns

## Summary
These examples demonstrate memory-efficiency techniques—coalesced and vectorized access, shared-memory tiling, bank-conflict avoidance, and async prefetch—to increase effective bandwidth.

## Performance Takeaways
- Convert uncoalesced into coalesced patterns to raise memory throughput
- Use vectorized loads/stores to saturate memory bandwidth more efficiently
- Tile into shared memory to increase reuse and arithmetic intensity
- Avoid bank conflicts for stable, predictable memory performance
- Use TMA/async prefetch to hide DRAM latency behind compute

This chapter contains code examples demonstrating GPU memory optimization techniques:

## Examples

### Memory Coalescing
- `uncoalesced_copy.cu` - Demonstrates uncoalesced memory access patterns
- `coalesced_copy.cu` - Optimized coalesced memory access
- `memory_access_pytorch.py` - PyTorch equivalent examples

### Vectorized Memory Access  
- `scalar_copy.cu` - Scalar memory operations (4-byte loads)
- `vectorized_copy.cu` - Vectorized operations using float4
- `vectorized_pytorch.py` - PyTorch vectorized operations

### Shared Memory Tiling
- `naive_matmul.cu` - Naive matrix multiplication with redundant global loads
- `tiled_matmul.cu` - Optimized matrix multiplication using shared memory tiling
- `matmul_pytorch.py` - PyTorch matrix multiplication examples

### Bank Conflict Avoidance
- `transpose_naive.cu` - Matrix transpose with bank conflicts
- `transpose_padded.cu` - Optimized transpose avoiding bank conflicts

### Read-Only Cache Optimization
- `naive_lookup.cu` - Standard global memory lookup
- `optimized_lookup.cu` - Using const __restrict__ for read-only cache

### Asynchronous Memory Prefetching
- `async_prefetch_tma.cu` - Using TMA and Pipeline API for async data movement

## Usage

Each example includes:
- CUDA C++ implementation
- PyTorch equivalent (where applicable) 
- Profiling commands for Nsight Compute analysis
- Performance comparison metrics

## Requirements

- CUDA 12.9+
- PyTorch 2.9+
- Nsight Compute for profiling
