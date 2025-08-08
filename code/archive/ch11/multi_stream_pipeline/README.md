# Multi-Stream Pipeline Example

This example demonstrates combining intra-kernel warp specialization with inter-kernel multi-stream pipelines.

## Features

- Combines warp-specialized kernels with multiple CUDA streams
- Uses stream-ordered memory allocation (`cudaMallocAsync`, `cudaFreeAsync`)
- Demonstrates three-way overlap: H2D copy, compute, D2H copy
- Shows how to feed multiple batches through the GPU simultaneously

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- Intra-kernel pipelining hides memory latency within each kernel
- Inter-kernel concurrency keeps multiple batches in flight
- Stream-ordered allocation prevents global synchronization
- Pinned host memory enables true asynchronous transfers
- This two-layer approach maximizes GPU utilization
