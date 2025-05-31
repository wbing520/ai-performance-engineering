# Chapter 6 Code Examples

This folder contains all of the CUDA C++ and PyTorch code snippets from Chapter 6 (Profiling and Tuning Memory Access Patterns), updated for:

- **CUDA 13**
- **PyTorch 2.7**
- **Blackwell B200** GPU

Each example has its own directory:

- **global_access/**: Uncoalesced vs. coalesced memory access (C++ & PyTorch)
- **vectorized_access/**: Scalar vs. vectorized loads (C++ & PyTorch)
- **shared_memory/**: Naive vs. padded transpose to avoid bank conflicts (C++)
- **tiling/**: Naive vs. tiled matrix multiply (C++ & PyTorch)
- **read_only_cache/**: Naive vs. __restrict__ lookup (C++)
- **async_tma/**: Example of cuda::memcpy_async with TMA (C++)

Profiler scripts are under **profiler_scripts/** for Nsight Systems (nsys) and Nsight Compute (ncu).

To build and run an example:

```bash
cd chapter6-examples/global_access
./run.sh

cd ../vectorized_access
./run.sh

# ...and so on for each directory.
```

For profiling:

```bash
bash profiler_scripts/nsys_profile.sh global_access/uncoalesced
bash profiler_scripts/ncu_profile.sh global_access/uncoalesced
```
