# Chapter 8 Code Examples

This folder contains all of the CUDA C++ and PyTorch code snippets from Chapter 8 (Synchronizing, Pipelining, and Overlapping Compute and Memory Transfers), updated for:

- **CUDA 13**
- **PyTorch 2.8**
- **Blackwell B200** GPU

Examples include:

- **intra_pipeline/**: Two-stage double-buffered GEMM pipeline using <cuda::pipeline>
- **warp_specialized/**: Warp-specialized pipeline kernel using <cuda::pipeline>
- **cooperative/**: Naive loop vs. persistent kernel (cooperative groups)
- **clusters/**: CTA-cluster example with DSMEM and cluster-wide barriers
- **streams/**: Overlapping compute and H2D/D2H transfers with streams and cudaMallocAsync
- **combined_pipeline/**: Streaming multiple mini-batches with a warp-specialized pipeline via streams

To build and run an example:

```bash
cd chapter8-examples/intra_pipeline && ./run.sh
cd ../warp_specialized && ./run.sh
cd ../cooperative && ./run.sh
cd ../clusters && ./run.sh
cd ../streams && ./run.sh
cd ../combined_pipeline && ./run.sh
```

#### Profiling

Profiler scripts are under **profiler_scripts/** for Nsight Systems (`nsys`) and Nsight Compute (`ncu`):

```bash
bash profiler_scripts/nsys_profile.sh intra_pipeline/gemm_tiled_pipeline
bash profiler_scripts/ncu_profile.sh intra_pipeline/gemm_tiled_pipeline
```
