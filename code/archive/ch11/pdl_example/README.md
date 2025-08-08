# Programmatic Dependent Launch (PDL) Example

This example demonstrates Programmatic Dependent Launch (PDL) for inter-kernel overlap.

## Features

- Shows how one kernel can trigger another kernel's execution directly on the device
- Uses `cudaTriggerProgrammaticLaunchCompletion()` and `cudaGridDependencySynchronize()`
- Demonstrates kernel overlap without CPU intervention
- Uses `cudaLaunchKernelEx()` with PDL attributes

## Building

```bash
make
```

## Running

```bash
make run
```

## Key Concepts

- PDL allows kernel B to begin execution before kernel A fully completes
- The trigger mechanism ensures memory flushes are complete before dependent work starts
- This technique masks kernel-launch overhead and maximizes GPU utilization
- PDL is particularly useful for complex pipelines with fine-grained dependencies
