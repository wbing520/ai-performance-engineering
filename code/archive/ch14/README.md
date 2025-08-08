# Chapter 14: PyTorch Compiler, XLA, and OpenAI Triton Backends

This chapter provides a deep dive into the PyTorch compilation stack including TorchDynamo, AOTAutograd, PrimTorch IR, and TorchInductor. It also covers OpenAI's Triton language for custom GPU kernel development and PyTorch XLA backend for non-NVIDIA accelerators.

## Overview

The chapter focuses on:
- **PyTorch Compiler Stack**: Understanding TorchDynamo, AOTAutograd, and TorchInductor
- **Graph Breaks**: Identifying and fixing compilation issues
- **Dynamic Shapes**: Handling variable input sizes
- **OpenAI Triton**: Writing custom GPU kernels in Python
- **Autotuning**: Automatic kernel optimization
- **Advanced Techniques**: Warp specialization, persistent kernels, pipelining
- **PyTorch XLA**: Alternative backend for non-NVIDIA hardware

## Files

### PyTorch Compiler Examples
- `torch_compiler_examples.py` - Comprehensive PyTorch compiler techniques and debugging

### Triton Kernel Development
- `triton_examples.py` - OpenAI Triton kernel development and optimization

## PyTorch Compiler Stack

### TorchDynamo (Graph Capture)
```python
# Basic compilation
compiled_model = torch.compile(model, mode="max-autotune")

# Force full graph compilation
compiled_model = torch.compile(model, fullgraph=True)
```

### Compilation Modes

| Mode | Description | Compile Time | Extra Memory | Features |
|------|-------------|--------------|--------------|----------|
| default | Balanced optimizations | Low-Medium | No | General fusion, basic autotuning |
| reduce-overhead | Reduces per-iteration overhead | Medium | Yes | Uses CUDA Graphs (if possible) |
| max-autotune | Maximizes runtime performance | High | Maybe | Aggressive Triton autotuning |
| max-autotune-no-cudagraphs | Same as max-autotune but without graphs | High | No | Same as above but disables graphs |

### Graph Breaks and Debugging

```python
# Analyze graph breaks
explanation = torch._dynamo.explain(model)(input)
print(f"Graph breaks: {explanation.graph_break_count}")

# Debug with logging
os.environ['TORCH_LOGS'] = 'graph_breaks,dynamo'
```

### Dynamic Shapes

```python
# Enable dynamic shapes
compiled_model = torch.compile(model, dynamic=True)

# Mark specific dimensions as dynamic
torch._dynamo.mark_dynamic(tensor, dim)
```

### Compiler Stances

```python
# Different compiler behaviors
torch.compiler.set_stance("default")           # Normal compilation
torch.compiler.set_stance("fail_on_recompile") # Error on recompile
torch.compiler.set_stance("eager_on_recompile") # Fallback to eager
torch.compiler.set_stance("force_eager")       # Always eager
```

## OpenAI Triton

### Basic Triton Kernel

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)
```

### Autotuning

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['n_elements']
)
@triton.jit
def autotuned_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Kernel implementation
    pass
```

### PyTorch Integration

```python
@triton_op("my_lib::vector_add", mutates_args=())
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    out = torch.empty_like(x)
    
    def grid_fn(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    
    wrap_triton(vector_add_kernel)[grid_fn](x, y, out, n, BLOCK_SIZE=1024)
    return out
```

### Shared Memory

```python
@triton.jit
def shared_memory_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Allocate shared memory
    shared_data = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Load data into shared memory
    x = tl.load(x_ptr + offsets, mask=mask)
    shared_data = x
    
    # Process in shared memory
    result = shared_data * 2.0
    tl.store(out_ptr + offsets, result, mask=mask)
```

### Persistent Kernels

```python
@triton.jit
def persistent_gemm_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    # Allocate shared memory buffers
    A_sh = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    B_sh = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in chunks
    for t in range(num_tiles):
        # Load tiles into shared memory
        A_sh = tl.load(a_ptrs)
        B_sh = tl.load(b_ptrs)
        
        # Compute partial matmul
        acc += tl.dot(A_sh, B_sh)
    
    # Write back result
    tl.store(c_ptrs, acc)
```

### Pipelined Kernels

```python
@triton.jit
def pipelined_matmul(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    # Pipelined loop with double-buffering
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
        # Load tiles (async for later stages)
        A_sh = tl.load(a_ptrs)
        B_sh = tl.load(b_ptrs)
        
        # Compute partial dot
        acc += tl.dot(A_sh, B_sh)
    
    # Write result
    tl.store(c_ptrs, acc)
```

### Warp Specialization

```python
@triton.jit
def warp_specialized_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Use warp specialization for the loop
    for i in tl.range(0, BLOCK_SIZE, num_stages=2, warp_specialize=True):
        idx = block_start + i
        if idx < n_elements:
            x = tl.load(x_ptr + idx)
            result = x * 2.0
            tl.store(out_ptr + idx, result)
```

### Tensor Cores

```python
@triton.jit
def wmma_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    # Use tl.dot which automatically uses Tensor Cores for FP16/BF16
    a = tl.load(A_ptr + ...)
    b = tl.load(B_ptr + ...)
    c = tl.dot(a, b)  # Automatically uses Tensor Cores
    tl.store(C_ptr + ..., c)
```

## Debugging and Optimization

### Environment Variables

```bash
# Enable various logging options
export TORCH_LOGS="graph_breaks,dynamo,inductor,perf_hints"
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_BENCHMARK_KERNEL=1
```

### Debugging Tools

```python
# Analyze graph breaks
explanation = torch._dynamo.explain(model)(input)

# Disable compiler temporarily
with torch._dynamo.disable():
    output = model(input)

# Allow function in graph
@torch._dynamo.allow_in_graph
def safe_function(x):
    return x * 2
```

### Performance Hints

```python
# Enable performance hints logging
os.environ['TORCH_LOGS'] = 'perf_hints'

# Common hints:
# - "CUDA graph not used because input is mutated"
# - "fell back to eager for random op"
# - "missed fusion opportunity"
```

## PyTorch XLA Backend

```python
# Use XLA backend for non-NVIDIA hardware
compiled_model = torch.compile(model, backend="openxla")

# XLA is optimized for static shapes
# May need to pad inputs to fixed sizes
```

## Key Techniques

### 1. Minimize Graph Breaks

```python
# Avoid Python control flow
# Instead of: if x.sum() > 0: y = f(x) else: y = g(x)
# Use: y = torch.where(x.sum() > 0, f(x), g(x))

# Avoid print statements during compilation
if not torch._dynamo.is_compiling():
    print("debug info")
```

### 2. Handle Dynamic Shapes

```python
# Mark dimensions as dynamic
torch._dynamo.mark_dynamic(tensor, dim)

# Use dynamic=True for variable input sizes
compiled_model = torch.compile(model, dynamic=True)
```

### 3. Optimize Triton Kernels

```python
# Use autotuning for optimal parameters
@triton.autotune(configs=[...])

# Use shared memory for data reuse
shared_data = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

# Use pipelining for memory bandwidth
for k in tl.range(0, K, num_stages=2):
    # async loads with compute overlap
```

### 4. Register Custom Operations

```python
@triton_op("my_lib::custom_op", mutates_args=())
def custom_op(x: torch.Tensor) -> torch.Tensor:
    # Implementation
    return result
```

## Performance Tips

1. **Start with torch.compile** - Often provides significant speedups with minimal effort
2. **Minimize graph breaks** - Use torch._dynamo.explain() to identify issues
3. **Use appropriate compilation modes** - Choose based on your workload
4. **Profile recompilations** - Monitor with TORCH_LOGS="guards,recompiles"
5. **Use dynamic shapes carefully** - Can disable CUDA graphs
6. **Write custom kernels only when needed** - Profile first to identify bottlenecks
7. **Use Triton autotuning** - Automatically find optimal parameters
8. **Leverage shared memory** - For data reuse patterns
9. **Implement pipelining** - For memory bandwidth optimization
10. **Use Tensor Cores** - With FP16/BF16 inputs

## Troubleshooting

### Common Issues

1. **Graph breaks**
   - Use torch._dynamo.explain() to identify causes
   - Replace Python control flow with tensor operations
   - Avoid print/logging during compilation

2. **Frequent recompilations**
   - Check for changing tensor shapes/dtypes
   - Use torch._dynamo.mark_dynamic() for known dynamic dimensions
   - Avoid Python random values in compiled code

3. **Memory issues**
   - Monitor memory usage with compiled kernels
   - Use smaller block sizes in Triton kernels
   - Consider compiling submodules separately

4. **Numerical differences**
   - Check for mixed precision usage
   - Use torch.set_float32_matmul_precision('high')
   - Enable deterministic algorithms if needed

### Debugging Commands

```bash
# Profile with Nsight Systems
nsys profile --output=profile python torch_compiler_examples.py

# Profile with Nsight Compute
ncu --kernel-name-regex "triton" python triton_examples.py

# Enable Triton debug mode
export TRITON_DEBUG=1
```

## Running the Examples

### Prerequisites
- PyTorch 2.8+
- CUDA 12.9+
- Triton library
- NVIDIA GPU with sufficient memory

### Basic Usage
```bash
# Run PyTorch compiler examples
python torch_compiler_examples.py

# Run Triton kernel examples
python triton_examples.py
```

### Environment Setup
```bash
# Enable debugging
export TORCH_LOGS="graph_breaks,dynamo,inductor"
export TORCH_COMPILE_DEBUG=1

# Enable Triton debugging
export TRITON_DEBUG=1
```

This chapter provides comprehensive tools for optimizing PyTorch models through compilation and custom kernel development. The examples demonstrate both the PyTorch compiler stack and OpenAI Triton for achieving maximum performance on modern GPU hardware.
