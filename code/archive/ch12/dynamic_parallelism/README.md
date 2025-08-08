# Dynamic Parallelism Examples

This example demonstrates Dynamic Parallelism (DP) for device-side kernel launching.

## Features

- **Host-Launched Version**: Shows traditional CPU-driven child kernel launches
- **Device-Launched Version**: Demonstrates GPU-initiated child kernel launches
- Compares performance and overhead between approaches

## Building

```bash
make
```

## Running

### Host-Launched Version
```bash
make run_host
```

### Device-Launched Version
```bash
make run_device
```

## Key Concepts

### Host-Launched Dynamic Parallelism
- Parent kernel runs, then CPU decides and launches child kernels
- Requires explicit `cudaDeviceSynchronize()` calls
- Creates idle gaps between parent and child execution
- 3 separate host launches (1 parent + 2 children)

### Device-Launched Dynamic Parallelism
- Parent kernel spawns child kernels directly on GPU
- Uses `cudaLaunchKernel()` for device-side launches
- Eliminates CPU-GPU coordination overhead
- Single host launch, children launched by parent

## Performance Benefits

- **Reduced Idle Time**: Eliminates CPU decision-making gaps
- **Better Data Locality**: Intermediate results stay in GPU memory
- **Lower Launch Overhead**: Fewer host-GPU handshakes
- **Improved Utilization**: Keeps SMs busy end-to-end

## Considerations

- Device launches have slightly higher per-launch overhead (~25µs vs ~20µs)
- Requires sufficient stack size (`cudaDeviceSetLimit`)
- Default limit of 2,048 pending child launches
- Best for irregular, data-dependent workloads
- Profile to ensure benefits outweigh overhead

## Use Cases

- Hierarchical reductions
- Adaptive mesh refinement
- Graph traversals
- Irregular algorithms where work emerges dynamically
