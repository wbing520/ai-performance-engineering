# Combined PDL + Thread Block Clusters + Warp Specialization Example

This example demonstrates the pinnacle of CUDA performance optimizations by combining three advanced techniques.

## Features

- **PDL (Programmatic Dependent Launch)**: Allows kernel overlap without CPU intervention
- **Thread Block Clusters**: Enables multi-SM cooperation and shared-memory multicast
- **Warp Specialization**: Subdivides work into producer/consumer warps within each block
- **TMA-like Async Copy**: Uses `cuda::memcpy_async` for efficient data movement

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- **Intra-Kernel Pipelining**: Warp specialization hides memory latency within each kernel
- **Inter-Kernel Overlap**: PDL allows dependent kernels to begin before previous kernels complete
- **Inter-Block Cooperation**: Thread block clusters coordinate work across multiple SMs
- **Complexity Trade-offs**: This combination provides maximum performance but requires careful synchronization

## Note

This example represents the most advanced CUDA optimization techniques. In practice, most applications achieve excellent performance with simpler combinations of these techniques.
