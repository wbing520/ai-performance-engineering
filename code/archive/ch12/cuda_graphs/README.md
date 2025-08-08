# CUDA Graph Examples

This example demonstrates CUDA Graph usage for reducing launch overhead in repetitive operations.

## Features

- Shows C++ CUDA Graph implementation
- Includes PyTorch CUDA Graph example
- Demonstrates graph capture and replay
- Compares performance with/without graphs

## Building

```bash
make
```

## Running

### C++ Version
```bash
make run
```

### PyTorch Version
```bash
make run_pytorch
```

## Key Concepts

- **Graph Capture**: Records a sequence of operations once using `cudaStreamBeginCapture`/`cudaStreamEndCapture`
- **Graph Replay**: Launches the entire sequence with a single `cudaGraphLaunch` call
- **Reduced Overhead**: Eliminates per-iteration CPU scheduling overhead
- **Memory Pools**: Static memory allocation prevents allocations from becoming part of the graph
- **Warm-up**: Required to initialize CUDA kernels and libraries before capture

## Performance Benefits

- 20-30% latency reduction for repetitive operations
- Eliminates per-kernel launch overhead
- Better GPU scheduling with known dependencies
- Reduced CPU-GPU coordination

## Considerations

- Memory addresses must remain fixed during graph execution
- Cannot include host callbacks or unsupported operations
- All tensors must be pre-allocated with fixed shapes
- Warm-up pass required to initialize lazy-loaded kernels
