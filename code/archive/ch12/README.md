# Chapter 12: Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration

This chapter covers advanced GPU orchestration techniques for maximizing utilization across single and multi-GPU systems.

## Code Examples

### Atomic Work Queue (`atomic_work_queue/`)
Demonstrates dynamic work distribution using L2-cache atomic counters with batching techniques to balance irregular workloads.

### CUDA Graphs (`cuda_graphs/`)
Shows how to capture and replay fixed operation sequences to reduce launch overhead, including both C++ and PyTorch implementations.

### Dynamic Parallelism (`dynamic_parallelism/`)
Compares host-launched vs device-launched child kernels, demonstrating how to eliminate CPU-GPU coordination overhead.

### NVSHMEM Example (`nvshmem_example/`)
Demonstrates fine-grained GPU-to-GPU communication using NVIDIA Shared Memory for one-sided remote memory operations.

## Key Concepts

- **Dynamic Scheduling**: Using atomic queues to balance irregular workloads across SMs
- **CUDA Graphs**: Capturing fixed pipelines to reduce per-iteration CPU overhead
- **Device-Initiated Launches**: Moving kernel orchestration to the GPU
- **Multi-GPU Communication**: Overlapping computation with peer-to-peer transfers
- **Roofline Analysis**: Using operational intensity to guide optimization choices

## Building and Running

Each example can be built and run independently:

```bash
cd atomic_work_queue
make
make run
```

## Dependencies

- CUDA 12.9+
- NVIDIA GPU with compute capability 9.0+
- C++17 support for advanced features
- NVSHMEM library (for NVSHMEM examples)
- MPI (for multi-GPU examples)

## Performance Considerations

- **Atomic Batching**: Reduces contention by claiming work in batches
- **Graph Capture**: 20-30% latency reduction for repetitive operations
- **Device Launches**: Eliminates CPU decision-making gaps
- **NVSHMEM**: Sub-microsecond latency for GPU-to-GPU communication
- **Roofline-Guided**: Choose optimizations based on memory vs compute bounds

## Orchestration Techniques

- **Memory-Bound**: Focus on overlap and reducing data movement
- **Compute-Bound**: Use launch-reduction techniques like fusion and graphs
- **Mixed Workloads**: Increase concurrency to utilize all hardware units
- **Multi-GPU**: Overlap communication with computation for linear scaling
