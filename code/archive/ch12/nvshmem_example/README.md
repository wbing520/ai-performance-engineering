# NVSHMEM Example

This example demonstrates NVIDIA Shared Memory (NVSHMEM) for fine-grained GPU-to-GPU communication.

## Features

- One-sided remote memory operations
- GPU-to-GPU direct communication without CPU intervention
- Sub-microsecond latency over NVLink
- Partitioned Global Address Space (PGAS) programming model

## Building

```bash
make
```

**Note**: You may need to adjust the NVSHMEM library paths in the Makefile based on your installation.

## Running

```bash
make run
```

This runs the example with 2 Processing Elements (PEs) using MPI.

## Key Concepts

### NVSHMEM Operations
- **`nvshmem_float_p()`**: One-sided put operation to write data to remote GPU
- **`nvshmem_quiet()`**: Ensures all previous RMA operations complete
- **`nvshmem_int_wait_until()`**: Wait for condition on remote memory
- **`nvshmem_barrier_all()`**: Global synchronization across all PEs

### Communication Pattern
1. **Sender (PE 0)**: Writes data to remote GPU memory, then signals completion
2. **Receiver (PE 1)**: Waits for signal, then processes the data
3. **No CPU Coordination**: All communication happens directly on GPUs

## Performance Benefits

- **Near Wire Speed**: Sub-microsecond latency over NVLink
- **Zero-Copy**: Direct GPU-to-GPU transfers without host staging
- **Low Overhead**: Bypasses CPU and software launch overhead
- **Fine-Grained**: Supports irregular, event-driven communication

## Use Cases

- Dynamic task queues
- Fine-grained event notifications
- Graph algorithms
- Discrete-event simulations
- Irregular communication patterns

## Considerations

- Requires careful synchronization to avoid races
- Over-synchronization can create bottlenecks
- Best for irregular, data-dependent workloads
- Can be captured in CUDA graphs for repetitive patterns
