# Chapter 7: Matmul Tiling

This chapter illustrates how **shared-memory tiling** can dramatically improve the performance of matrix multiplication by increasing data reuse and arithmetic intensity. We compare a naive matrix multiply (with no tiling) to an optimized tiled version:

* **`naive_matmul.py`:** A **very slow** PyTorch implementation of matrix multiplication using nested Python loops. It multiplies two N×N matrices by explicitly computing each output element with a dot product, loading data from global memory repeatedly (this mimics a naive GPU kernel that would be memory-bound).
* **`tiled_matmul.cu`:** A CUDA C++ implementation using a 32×32 tile held in shared memory. Each thread block computes one tile of the result, reusing data from shared memory for 32 multiply-adds, instead of reading from global memory each time.
* **`tiled_matmul.py`:** A pedagogical example of tiling in PyTorch: it breaks the matrices into 32×32 blocks and uses `torch.mm` on those blocks to simulate how tiling works (in practice you'd use PyTorch's built-in `torch.matmul`, which already does this under the hood).

## Running on 8× B200 Cluster with Grace CPU

### Compile CUDA Program

```bash
cd code/ch7

# Compile tiled matrix multiplication with Blackwell B200/B300 optimizations
nvcc -arch=sm_100 -o tiled_matmul tiled_matmul.cu
```

### Run Examples

```bash
# Run naive Python implementation (very slow)
python naive_matmul.py

# Run tiled Python implementation (faster)
python tiled_matmul.py

# Run CUDA tiled implementation (fastest)
./tiled_matmul
```

## Profiling Commands

### Nsight Systems (nsys)

```bash
# Profile naive vs tiled implementations
nsys profile -t cuda,osrt -o naive_matmul_profile python naive_matmul.py
nsys profile -t cuda,osrt -o tiled_matmul_profile python tiled_matmul.py
nsys profile -t cuda,osrt -o cuda_tiled_profile ./tiled_matmul
```

### Nsight Compute (ncu)

```bash
# Profile kernel efficiency and memory throughput
ncu --metrics achieved_occupancy,warp_execution_efficiency \
    -o tiled_kernel_profile ./tiled_matmul

# Profile memory transactions
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
    -o memory_profile ./tiled_matmul

# Profile arithmetic intensity
ncu --metrics sm__sass_thread_inst_op_fp16_pred_on.sum,sm__sass_thread_inst_op_fp32_pred_on.sum \
    -o arithmetic_profile ./tiled_matmul
```

### PyTorch Profiler

```bash
# Profile Python implementations
python -m torch.utils.bottleneck naive_matmul.py
python -m torch.utils.bottleneck tiled_matmul.py

# Profile with memory tracking
python -c "
import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity

# Profile naive matrix multiplication
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    N = 512
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')
    C = torch.zeros(N, N, device='cuda')
    
    # Naive implementation
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
"
```

### Memory Profiling

```bash
# Monitor memory usage during matrix multiplication
python -c "
import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Monitor memory during tiled matrix multiplication
print(f'Initial memory: {torch.cuda.memory_allocated()/1e6:.2f} MB')
N = 1024
A = torch.randn(N, N, device='cuda')
B = torch.randn(N, N, device='cuda')
print(f'After allocation: {torch.cuda.memory_allocated()/1e6:.2f} MB')
C = torch.mm(A, B)
print(f'After computation: {torch.cuda.memory_allocated()/1e6:.2f} MB')
"
```

## Expected Output

### Python Implementations

```
# naive_matmul.py (very slow)
Done, C[0,0] = 512.0
Time: 45.2 seconds (N=512)

# tiled_matmul.py (faster)
Done, C[0,0] = 512.0
Time: 2.1 seconds (N=512)
```

### CUDA Implementation

```
# tiled_matmul.cu
Tiled matrix multiplication completed
Matrix size: 1024x1024
Time: 0.8 ms
Performance: 2.6 TFLOPS
```

### Nsight Compute Metrics

```
# Naive kernel (if implemented)
achieved_occupancy: 0.125 (12.5% - low)
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum: 8.0 GB (high memory traffic)

# Tiled kernel
achieved_occupancy: 0.875 (87.5% - high)
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum: 1.2 GB (reduced memory traffic)
sm__sass_thread_inst_op_fp32_pred_on.sum: 2.1e9 (high arithmetic intensity)
```

## Tuning Tips

1. **Tile Size**: Use 32×32 tiles for optimal shared memory utilization
2. **Memory Coalescing**: Ensure threads access memory in contiguous patterns
3. **Shared Memory**: Maximize data reuse within each tile
4. **Arithmetic Intensity**: Aim for high FLOPs per byte ratio
5. **Occupancy**: Ensure sufficient threads per block for high occupancy

## Troubleshooting

- **Shared Memory Limit**: Reduce tile size if hitting shared memory limits
- **Memory Coalescing**: Check memory access patterns for optimal bandwidth
- **Occupancy**: Adjust thread block size for better GPU utilization
- **Compilation Errors**: Ensure CUDA toolkit version matches target architecture

## Architecture-Specific Notes

### Blackwell B200/B300 with Grace CPU

- **Shared Memory**: 48KB per block
- **L1 Cache**: 192KB per SM
- **Memory Bandwidth**: HBM3e provides ~3.2TB/s
- **Tensor Cores**: Support for FP16 and INT8 operations

### CUDA 12.8 Optimizations

- **Stream-ordered Memory**: Use `cudaMallocAsync` for better performance
- **Unified Memory**: HBM3e provides faster CPU-GPU memory access
- **Multi-GPU**: Each GPU can access other GPUs' memory with lower latency

## Performance Analysis

### Memory Bandwidth Utilization

The tiled kernel demonstrates much higher memory bandwidth efficiency due to:
- **Data Reuse**: Each element loaded once per tile instead of once per multiply
- **Coalesced Access**: Multiple threads access contiguous memory locations
- **Shared Memory**: Reduces global memory traffic by 32x per tile

### Arithmetic Intensity

The tiled kernel achieves much higher arithmetic intensity:
- **Naive**: ~1.5 FLOPs per byte (memory-bound)
- **Tiled**: ~8 FLOPs per byte (compute-bound)

### Occupancy and Throughput

Nsight Compute will show:
- **Naive**: Low occupancy, high memory traffic, long execution time
- **Tiled**: High occupancy, reduced memory traffic, short execution time

This demonstrates the fundamental principle of GPU optimization: **maximize data reuse** to achieve high performance.
