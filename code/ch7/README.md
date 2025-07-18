# Chapter 7 Code Examples

This folder contains all of the CUDA C++ and PyTorch code snippets from Chapter 7 (Profiling, Tuning, and Increasing Arithmetic Intensity), updated for:

- **CUDA 13**
- **PyTorch 2.7**
- **Blackwell B200** GPU

Examples extracted from Chapter 7: Profiling, Tuning, and Increasing Arithmetic Intensity

Each example has its own directory. From the repo root you can:

```bash
# Threshold naive vs predicated
cd threshold && ./run.sh

# PyTorch threshold operations
cd threshold_py && ./run.sh

# Independent ILP ops
cd independent_ilp && ./run.sh

# Naive FP32 GEMM
cd gemm_fp32 && ./run.sh

# Tensor Core WMMA GEMM
cd gemm_tensorcore && ./run.sh

# Fused L2 Norm example
cd fused_l2norm && ./run.sh

# CUTLASS GEMM example
cd cutlass_gemm && ./run.sh
```

#### Profiling

Profiler scripts are under **profiler_scripts/** for Nsight Systems (`nsys`) and Nsight Compute (`ncu`):

```bash
bash profiler_scripts/nsys_profile.sh gemm_fp32/matmul_fp32
bash profiler_scripts/ncu_profile.sh gemm_fp32/matmul_fp32
```
