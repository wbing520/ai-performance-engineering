# Chapter 9: Kernel Fusion and Arithmetic Intensity

This chapter focuses on kernel fusion techniques to improve arithmetic intensity and reduce memory bandwidth requirements in AI workloads.

## Code Examples

### CUDA Kernel Fusion
- `fused_l2norm.cu` - Basic fused L2 normalization kernel example
- `fusedL2Norm.cu` - Advanced fused L2 normalization with optimizations and benchmarking
- `cutlass_gemm_example.cu` - CUTLASS GEMM example for optimal arithmetic intensity
- `inline_ptx_example.cu` - Inline PTX examples for micro-optimizations

### PyTorch Fusion
- `fusion_pytorch.py` - PyTorch kernel fusion examples using torch.compile

## Key Concepts

### Arithmetic Intensity
- **Definition**: Ratio of arithmetic operations to memory operations
- **Goal**: Maximize compute utilization relative to memory bandwidth
- **Impact**: Higher arithmetic intensity → better GPU utilization

### Fusion Benefits
1. **Reduced Memory Traffic**: Eliminate intermediate results stored to global memory
2. **Improved Cache Locality**: Data stays in cache between operations
3. **Lower Kernel Launch Overhead**: Fewer kernel launches
4. **Better Occupancy**: More work per thread block

### Fusion Strategies

#### Manual CUDA Fusion
```cuda
// Instead of separate kernels:
__global__ void kernel1(float* in, float* temp, int N);
__global__ void kernel2(float* temp, float* out, int N);

// Use single fused kernel:
__global__ void fused_kernel(float* in, float* out, int N) {
    // Compute both operations in single kernel
}
```

#### PyTorch torch.compile Fusion
```python
@torch.compile
def fused_operations(x):
    # Multiple operations fused automatically
    return F.gelu(F.layer_norm(F.linear(x, weight, bias), [x.size(-1)]))
```

## Performance Optimizations

### Memory Access Patterns
- Coalesced global memory access
- Effective use of shared memory
- Minimize global memory round-trips

### Computational Optimizations
- Instruction-level parallelism (ILP)
- Warp-level primitives
- Tensor Core utilization

### Occupancy Tuning
- Launch bounds optimization
- Register usage optimization
- Shared memory usage optimization

## Building and Running

### CUDA Examples
```bash
make all                    # Build all CUDA examples
./fusedL2Norm 128 4096 100 # Run L2 norm benchmark
./cutlass_gemm_example     # Run CUTLASS GEMM example
./inline_ptx_example       # Run inline PTX examples
```

### PyTorch Examples
```bash
python fusion_pytorch.py   # Run PyTorch fusion benchmark
```

## Profiling

### Nsight Compute
```bash
# Profile memory bandwidth and arithmetic intensity
ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis ./fusedL2Norm

# Profile instruction mix and cache usage
ncu --section InstructionStats --section LaunchStats ./inline_ptx_example
```

### Nsight Systems
```bash
# Profile kernel fusion effectiveness
nsys profile --force-overwrite=true -o fusion_analysis ./fusedL2Norm
```

### PyTorch Profiler
```python
import torch.profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    fused_operations(x)
prof.export_chrome_trace("fusion_trace.json")
```

## Key Metrics

### Arithmetic Intensity Calculation
```
AI = (FLOPs) / (Bytes transferred)
```

### Target Metrics
- **L2 Norm**: ~3-5x speedup with fusion
- **Attention**: ~2-3x speedup with compilation
- **Memory Reduction**: 20-40% with intermediate elimination

## Hardware Considerations

### GPU Architecture Impact
- **Compute Capability**: Affects available optimizations
- **Memory Hierarchy**: L1/L2 cache sizes impact fusion benefits  
- **Tensor Cores**: Specialized units for mixed-precision operations

### CUTLASS Integration
- Optimized for specific GPU architectures
- Automatic kernel selection based on problem size
- Support for various precision formats (FP16, INT8, etc.)

## Best Practices

1. **Profile First**: Identify memory-bound kernels
2. **Fuse Related Operations**: Group operations with data dependencies
3. **Balance Work**: Ensure sufficient work per fused kernel
4. **Consider Precision**: Mixed-precision can improve arithmetic intensity
5. **Validate Correctness**: Ensure fusion doesn't change numerical results

## Requirements

- CUDA 12.9+
- PyTorch 2.8+
- CUTLASS (for CUTLASS examples)
- Nsight Compute/Systems for profiling