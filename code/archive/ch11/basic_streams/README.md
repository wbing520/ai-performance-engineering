# Basic CUDA Streams Example

This example demonstrates basic CUDA stream usage for overlapping kernel execution.

## Features

- Creates two CUDA streams
- Launches 5 kernels across the two streams
- Shows CPU-GPU overlap
- Demonstrates stream synchronization

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- CUDA streams allow overlapping kernel execution
- CPU can continue working while GPU kernels run asynchronously
- Stream synchronization ensures all work completes before cleanup
