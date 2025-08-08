# Stream-Ordered Memory Allocator Example

This example demonstrates the use of CUDA's stream-ordered memory allocator for asynchronous memory management.

## Features

- Uses `cudaMallocAsync` and `cudaFreeAsync` for stream-ordered allocation
- Demonstrates memory pool configuration
- Shows overlap between memory allocation, computation, and data transfers
- Uses pinned host memory for true asynchronous transfers

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- Stream-ordered allocator avoids global device synchronization
- Memory allocations are enqueued in the stream's operation queue
- Pinned host memory enables true asynchronous transfers
- Memory pool configuration controls when memory is returned to the OS
