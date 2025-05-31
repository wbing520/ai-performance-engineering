# Chapter 8 Code Examples

This folder contains all of the CUDA C++ and PyTorch code snippets from Chapter 8 (Synchronizing, Pipelining, and Overlapping Compute and Memory Transfers), updated for:

- **CUDA 13**
- **PyTorch 2.7**
- **Blackwell B200** GPU

Examples extracted from Chapter 8: advanced execution models, intra-kernel pipelining, CUDA Pipeline API, cooperative groups, persistent kernels, CTA clusters, and CUDA streams for inter-kernel concurrency fileciteturn3file0

Each example has its own directory. From the repo root you can:

```bash
# Double-buffered GEMM pipeline
cd intra_pipeline && ./run.sh

# Warp-specialized pipeline kernel
cd warp_specialized && ./run.sh

# Cooperative vs persistent kernels
cd cooperative && ./run.sh

# CTA cluster example
cd clusters && ./run.sh

# CUDA streams overlap example
cd streams && ./run.sh

# Combined pipeline with streams + cudaMallocAsync + cp.async
cd combined_pipeline && ./run.sh
```

#### Profiling

Profiler scripts are under **profiler_scripts/** for Nsight Systems (`nsys`) and Nsight Compute (`ncu`):

```bash
bash profiler_scripts/nsys_profile.sh intra_pipeline/gemm_tiled_pipeline
bash profiler_scripts/ncu_profile.sh intra_pipeline/gemm_tiled_pipeline
```
