# Chapter 12: Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration

## Summary
These examples demonstrate dynamic work distribution and CUDA Graphs (including device‑initiated launch) to reduce CPU overhead and maintain balanced, continuous GPU execution.

## Performance Takeaways
- Use dynamic work queues to balance irregular workloads across SMs
- Capture/replay with CUDA Graphs to remove per‑kernel launch overhead
- Launch graphs from device to eliminate CPU round‑trips in critical loops
- Update/conditional nodes to adapt graphs without costly recapture
- Combine queues + graphs for up to ~3× speedups on complex pipelines

This chapter focuses on advanced GPU orchestration techniques including dynamic work queues, CUDA graphs for reducing launch overhead, and device-initiated kernel launches that eliminate CPU involvement in scheduling decisions.

## Code Examples

### Dynamic Work Distribution
- `atomic_work_queue.cu` - Atomic counters and dynamic work queues for load balancing
- `cuda_graphs.cu` - CUDA graphs for batching operations and reducing launch overhead
- `dynamic_parallelism.cu` - Device-initiated kernel launches and adaptive scheduling

## Key Concepts

### Dynamic Work Queues with Atomic Counters
Load imbalance wastes compute resources when some threads finish early while others continue working. Dynamic work distribution solves this using atomic counters.

**Traditional Static Assignment Problems:**
- Fixed thread-to-work mapping leads to idle SMs
- Variable work per thread causes load imbalance
- Poor GPU utilization and wasted cycles

**Dynamic Work Queue Solution:**
```cuda
__device__ unsigned int globalIndex = 0;

__global__ void dynamicWorkKernel(float* data, int N) {
    while (true) {
        // Warp leader claims next batch
        unsigned int base;
        if (lane == 0) {
            base = atomicAdd(&globalIndex, warpSize);
        }
        base = __shfl_sync(mask, base, 0);
        
        unsigned int idx = base + lane;
        if (idx >= N) break;
        
        // Process work...
    }
}
```

**Performance Benefits:**
- 2x speedup for severely imbalanced workloads
- 10-20% improvement for moderate imbalance
- Near-uniform SM utilization
- Automatic load balancing

### CUDA Graphs
CUDA graphs eliminate per-kernel launch overhead by capturing and replaying entire workflows.

**Graph Capture and Replay:**
```cuda
// Capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernelA<<<grid, block, 0, stream>>>(data);
kernelB<<<grid, block, 0, stream>>>(data);
kernelC<<<grid, block, 0, stream>>>(data);
cudaStreamEndCapture(stream, &graph);

// Instantiate and replay
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
for (int i = 0; i < iterations; ++i) {
    cudaGraphLaunch(graphExec, stream); // Single launch for all kernels!
}
```

**Performance Improvements:**
- 25-50% reduction in launch overhead
- Better GPU scheduling with known dependencies
- Elimination of CPU-GPU handshakes
- Continuous kernel execution without gaps

### Graph Update and Conditional Nodes
Dynamic graph updates avoid recapture costs for parameter changes:

```cuda
// Update existing graph parameters
cudaGraphExecKernelNodeSetParams(graphExec, node, &newParams);

// Conditional execution based on device-computed values
cudaGraphConditionalHandle handle;
cudaGraphSetConditional(handle, flag); // Set by device kernel
```

### Device-Initiated Graph Launch
Remove CPU from scheduling decisions entirely:

```cuda
__global__ void persistentScheduler(...) {
    while (hasWork()) {
        if (condition) {
            // Launch graph directly from device
            cudaGraphLaunch(graphExec, cudaStreamGraphTailLaunch);
        }
    }
}
```

**Modes:**
- **Fire-and-forget**: Immediate concurrent execution
- **Tail launch**: Execute after current kernel completes
- **Sibling**: Run as peer in parent's stream environment

### Dynamic Parallelism
Device kernels can launch child kernels based on runtime conditions:

```cuda
__global__ void parentKernel(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (needsProcessing(data[idx])) {
        // Launch child kernel from device
        childKernel<<<grid, block>>>(data + idx, size);
    }
}
```

## Performance Optimization Strategies

### Atomic Counter Optimization
1. **Batch Work**: Use warp-level batching to reduce atomic frequency
2. **Hierarchical Distribution**: Block-level then warp-level batching
3. **Monitor Contention**: Track atomic_transactions_per_request in Nsight Compute

### CUDA Graph Best Practices
1. **Capture Strategy**: Capture with maximum expected size, then update
2. **Memory Management**: Use static pools to avoid allocations in graphs
3. **Warm-up**: Run operations once before capture to initialize libraries
4. **Conditional Nodes**: Use for device-side branching without CPU

### Device Launch Optimization
1. **Graph Residency**: Upload graphs to device memory for faster launch
2. **Launch Modes**: Choose appropriate mode (fire-and-forget vs tail)
3. **Resource Management**: Monitor kernel nesting depth and memory usage

## Building and Running

### Compilation
```bash
make all                    # Build all examples
make atomic_work_queue      # Build work queue example
make cuda_graphs           # Build graphs example  
make dynamic_parallelism   # Build dynamic parallelism example
```

### Execution
```bash
# Dynamic work distribution benchmark
./atomic_work_queue

# CUDA graphs demonstration
./cuda_graphs

# Device-initiated kernel orchestration
./dynamic_parallelism
```

## Performance Analysis

### Key Metrics

#### Work Queue Efficiency
- **SM Utilization**: Target >90% with dynamic distribution
- **Load Imbalance**: Measure variance in thread completion times
- **Atomic Contention**: Monitor transactions_per_request ratio

#### Graph Performance
- **Launch Overhead**: Compare traditional vs graph launch times
- **CPU-GPU Gaps**: Eliminate idle periods between kernels
- **Update Efficiency**: Microsecond graph parameter updates

#### Dynamic Parallelism
- **Kernel Nesting**: Monitor depth and resource usage
- **Launch Patterns**: Analyze child kernel creation patterns
- **Occupancy Impact**: Ensure parent kernels don't starve resources

### Expected Speedups
- **Dynamic Queues**: 1.5-2x for imbalanced workloads
- **CUDA Graphs**: 1.25-1.5x for multi-kernel pipelines
- **Device Graphs**: 2x faster launch than host-initiated
- **Combined**: Up to 3x improvement for complex workflows

## Profiling and Debugging

### Nsight Compute Analysis
```bash
# Atomic contention analysis
ncu --section MemoryWorkloadAnalysis --section WarpStateStats ./atomic_work_queue

# Graph launch analysis
ncu --section LaunchStats ./cuda_graphs

# Dynamic parallelism analysis
ncu --section LaunchStats --section WarpStateStats ./dynamic_parallelism
```

### Nsight Systems Timeline
```bash
# Work distribution patterns
nsys profile --force-overwrite=true -o work_queue ./atomic_work_queue

# Graph replay efficiency
nsys profile --force-overwrite=true -o graphs ./cuda_graphs

# Kernel hierarchy visualization
nsys profile --force-overwrite=true -o dynamic ./dynamic_parallelism
```

### Key Patterns to Look For
- **Uniform Execution**: Even SM utilization with dynamic queues
- **Continuous Kernels**: No gaps between graph-launched kernels
- **Nested Launches**: Proper child kernel creation in timelines
- **Reduced Overhead**: Fewer host API calls with graphs

## Hardware Requirements

### Compute Capability Requirements
- **Atomic Operations**: All modern GPUs (CC 2.0+)
- **CUDA Graphs**: CC 7.5+ (Turing, Ampere, Blackwell)
- **Dynamic Parallelism**: CC 3.5+ (Kepler and newer)
- **Device Graph Launch**: CC 7.5+ with recent CUDA toolkit

### Memory and Resources
- **L2 Cache**: Fast atomics require adequate L2 (64MB+ recommended)
- **Kernel Slots**: Up to 128 concurrent kernels on modern GPUs
- **Memory Pools**: Stream-ordered allocation for graph efficiency
- **SM Resources**: Shared among parent and child kernels

## Integration with AI Frameworks

### PyTorch CUDA Graphs
```python
# Enable graph capture in PyTorch
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)

# Replay for inference
for batch in dataloader:
    g.replay()  # Fast execution without overhead
```

### TensorRT and Inference Engines
- Pre-capture graphs for different batch sizes
- Use conditional nodes for dynamic batching
- Device-initiated graphs for continuous inference pipelines

## Common Pitfalls and Solutions

### Dynamic Queues
**Problem**: High atomic contention
**Solution**: Increase batch size, use hierarchical distribution

**Problem**: Load imbalance persists
**Solution**: Profile work distribution, adjust granularity

### CUDA Graphs
**Problem**: Graph capture fails
**Solution**: Avoid unsupported operations (malloc, host callbacks)

**Problem**: Memory address changes
**Solution**: Use static memory pools, update graph parameters

### Dynamic Parallelism
**Problem**: Performance regression
**Solution**: Profile overhead, consider alternatives like conditional graphs

**Problem**: Resource exhaustion
**Solution**: Limit nesting depth, monitor kernel slots

## Advanced Techniques

### Combined Optimizations
1. **Atomic Queues + Graphs**: Dynamic work with graph efficiency
2. **Persistent + Device Launch**: GPU-resident schedulers
3. **Conditional + Dynamic**: Adaptive execution patterns

### Multi-GPU Orchestration
- Device-to-device graph launches
- Peer-to-peer memory access in graphs
- NCCL communication nodes in graphs

### Custom Schedulers
- GPU-resident task schedulers
- Priority-based work distribution
- Adaptive algorithm selection

## Future Directions

### Hardware Evolution
- Improved atomic performance in newer architectures
- Enhanced graph capabilities (more conditional types)
- Better debugging tools for complex orchestration

### Software Advances
- Framework integration improvements
- Automatic graph optimization
- AI-driven scheduling decisions

## Requirements

- CUDA 11.0+ (CUDA 12.9+ recommended for all features)
- Compute Capability 3.5+ (dynamic parallelism)
- Compute Capability 7.5+ (CUDA graphs, device launch)
- Adequate L2 cache for atomic performance
- Modern GPU with concurrent kernel support