# Chapter 5 Code Examples

This folder contains all of the CUDA C++ and PyTorch code snippets from Chapter 5, updated for:

- **CUDA 13**  
- **PyTorch 2.7**  
- **Blackwell B200** GPU

Each example has its own directory. From the repo root you can:

```bash
# Build & run the simple 1D doubling kernel
cd simple_kernel && ./run.sh

# Build & run the 2D kernel
cd ../kernel_2d && ./run.sh

# Build & run the async alloc example
cd ../async_alloc && ./run.sh

# Build & run the unified memory example
cd ../unified_memory && ./run.sh

# Build & profile the vector-add examples
cd ../add_examples && ./profiler.sh
```

#### Profiling

We provide two generic scripts in `profiler_scripts/`:

- **nsys_profile.sh** – runs Nsight Systems to capture a timeline  
- **ncu_profile.sh** – runs Nsight Compute to gather per-kernel metrics (occupancy, warp-execution efficiency, memory‑pipe utilization, etc.)

Example:

```bash
bash profiler_scripts/nsys_profile.sh simple_kernel/simple_kernel
bash profiler_scripts/ncu_profile.sh simple_kernel/simple_kernel
```

For the PyTorch examples we also show how to use `torch.profiler` (in `add_parallel.py`) to capture CPU+GPU traces and inspect them in TensorBoard.
