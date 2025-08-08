# Warp-Specialized Pipeline Example

This example demonstrates intra-kernel pipelining using warp specialization and the CUDA Pipeline API.

## Features

- Uses three warps per thread block: loader, compute, and storer
- Implements a 3-stage pipeline using `cuda::pipeline`
- Demonstrates asynchronous memory operations with `cuda::memcpy_async`
- Shows how to hide memory latency within a single kernel

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- Warp specialization divides work across different warps within a thread block
- The CUDA Pipeline API provides fine-grained synchronization without full-block barriers
- Asynchronous memory operations allow overlap between data movement and computation
- This technique maximizes SM utilization by keeping all hardware units busy
