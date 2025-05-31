# Chapter 9 Code Examples

This folder contains all of the CUDA C++ and PyTorch code snippets from Chapter 9 (Dynamic Multi-Kernel Orchestration and Graph Scheduling), updated for:

- **CUDA 13**
- **PyTorch 2.7**
- **Blackwell B200** GPU

Each example has its own directory. From the repo root you can:

```bash
cd atomic_queue && ./run.sh
cd graphs && ./run.sh
cd dp && ./run.sh
```

#### Profiling

Profiler scripts are under **profiler_scripts/** for Nsight Systems (`nsys`) and Nsight Compute (`ncu`):

```bash
bash profiler_scripts/nsys_profile.sh atomic_queue/uneven_dynamic
bash profiler_scripts/ncu_profile.sh atomic_queue/uneven_dynamic
```
