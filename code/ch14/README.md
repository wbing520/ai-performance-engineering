# Chapter 14: PyTorch Compiler and Triton Programming

This directory contains comprehensive examples for PyTorch 2.8's torch.compile optimization and OpenAI's Triton 3.4 for custom GPU kernel development.

## Files Overview

### Core Examples

- **`torch_compiler_examples.py`** - PyTorch compiler optimization patterns, graph break analysis, and compilation modes
- **`triton_examples.py`** - Custom GPU kernel development with Triton, including vector operations, matrix multiplication, and fused kernels
- **`requirements.txt`** - Dependencies including PyTorch 2.8, Triton 3.4, and profiling tools

## Key Technologies Covered

### PyTorch Compiler (torch.compile)

#### Compilation Modes
- **`default`**: Balanced optimization for most workloads
- **`reduce-overhead`**: Minimize Python overhead for small models
- **`max-autotune`**: Aggressive optimization for maximum performance

#### Graph Break Analysis
- Debug tool: `torch._dynamo.explain()` 
- Common causes: side effects, dynamic control flow, unsupported operations
- Solutions: `torch.cond()`, `torch.where()`, compilation guards

#### Dynamic Shape Support
- Enable with `dynamic=True` for variable input sizes
- Automatic recompilation for new shapes
- Performance considerations for shape variations

### Triton GPU Programming

#### Core Concepts
- **SPMD Model**: Single Program Multiple Data execution
- **Block-based Programming**: Vectorized operations on data blocks
- **Memory Hierarchy**: Automatic shared memory management
- **Kernel Registration**: Integration with PyTorch operations

#### Kernel Types Demonstrated
1. **Vector Operations**: Element-wise computations
2. **Matrix Multiplication**: Tiled GEMM with shared memory
3. **Fused Activations**: Combined operations for efficiency
4. **Attention Kernels**: Simplified Flash Attention implementation

## Usage Examples

### PyTorch Compiler Optimization

```bash
# Run compiler examples with different modes
python torch_compiler_examples.py

# Expected output:
# - Graph break analysis
# - Performance benchmarks across compilation modes
# - Dynamic shape handling examples
```

### Triton Custom Kernels

```bash
# Test and benchmark Triton kernels
python triton_examples.py

# Expected output:
# - Kernel correctness verification
# - Performance comparison vs PyTorch built-ins
# - Speedup measurements
```

### Interactive Usage

```python
import torch
from torch_compiler_examples import SimpleModel, benchmark_compilation_modes
from triton_examples import vector_add_triton, matmul_triton

# Compile a model for optimization
model = SimpleModel()
compiled_model = torch.compile(model, mode='max-autotune')

# Use custom Triton kernels
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')
result = vector_add_triton(x, y)
```

## Performance Optimizations

### torch.compile Best Practices

1. **Avoid Graph Breaks**
   ```python
   # Bad: Side effects during computation
   def model(x):
       print(f"Processing {x.shape}")  # Causes graph break
       return x * 2
   
   # Good: Guard side effects
   def model(x):
       if not torch._dynamo.is_compiling():
           print(f"Processing {x.shape}")
       return x * 2
   ```

2. **Use torch.cond for Dynamic Control Flow**
   ```python
   # Bad: Data-dependent branching
   def model(x):
       if x.sum() > 0:
           return x * 2
       else:
           return x * -1
   
   # Good: Structured control flow
   def model(x):
       def true_fn(x): return x * 2
       def false_fn(x): return x * -1
       return torch.cond(x.sum() > 0, true_fn, false_fn, (x,))
   ```

3. **Enable Dynamic Shapes When Needed**
   ```python
   # For variable input sizes
   compiled_model = torch.compile(model, dynamic=True)
   ```

### Triton Optimization Patterns

1. **Memory Coalescing**
   ```python
   # Ensure contiguous memory access patterns
   offsets = block_start + tl.arange(0, BLOCK_SIZE)
   data = tl.load(ptr + offsets, mask=mask)
   ```

2. **Shared Memory Utilization**
   ```python
   # Allocate shared memory for tile storage
   a_shared = tl.shared_memory((BLOCK_M, BLOCK_K), dtype)
   ```

3. **Kernel Fusion**
   ```python
   # Combine multiple operations in single kernel
   # Example: ReLU + Dropout + LayerNorm in one pass
   ```

## Performance Benchmarks

### Expected Speedups (PyTorch Compiler)

- **Simple models**: 1.2-2x speedup over eager execution
- **Complex models**: 2-5x speedup with proper optimization
- **Memory-bound workloads**: 1.5-3x improvement
- **Compute-bound workloads**: 3-10x acceleration

### Expected Performance (Triton)

- **Vector operations**: Competitive with PyTorch built-ins
- **Matrix multiplication**: 80-95% of cuBLAS performance
- **Fused kernels**: 2-5x speedup over separate operations
- **Custom attention**: Comparable to Flash Attention

## Integration Examples

### Combining torch.compile with Triton

```python
@triton_op("mylib::custom_kernel") 
def custom_triton_op(x):
    return custom_triton_kernel(x)

@torch.compile
def optimized_model(x):
    # PyTorch operations get compiled
    x = torch.relu(x)
    
    # Custom Triton operation is included in graph
    x = custom_triton_op(x)
    
    return x.sum()
```

### Memory-Efficient Patterns

```python
class OptimizedModel(torch.nn.Module):
    def forward(self, x):
        # Use gradient checkpointing for memory efficiency
        x = torch.utils.checkpoint.checkpoint(self.layer1, x)
        
        # Apply torch.compile optimization
        x = self.compiled_layer2(x)
        
        return x

# Compile specific layers
model.compiled_layer2 = torch.compile(model.layer2, mode='reduce-overhead')
```

## Hardware Requirements

- **GPU**: NVIDIA GPUs with CUDA Compute Capability 7.0+
- **CUDA**: Version 12.9 or later
- **Memory**: Minimum 8GB GPU memory for larger examples
- **PyTorch**: Version 2.8 with CUDA support
- **Triton**: Version 3.4 compatible with PyTorch 2.8

## Profiling and Debugging

### Debugging Graph Breaks

```python
# Analyze compilation issues
explanation = torch._dynamo.explain(model)(sample_input)
print(explanation)  # Shows break reasons and locations
```

### Performance Profiling

```python
# Profile compiled models
with torch.profiler.profile() as prof:
    compiled_model(x)

print(prof.key_averages().table())
```

### Triton Kernel Debugging

```bash
# Enable debug output
export TRITON_DEBUG=1

# Profile kernel performance
nsys profile python triton_examples.py
```

## Troubleshooting

### Common Issues

1. **Slow Compilation Time**
   - Use `mode='reduce-overhead'` for development
   - Cache compiled models for repeated use
   - Consider compilation overhead vs runtime speedup

2. **Graph Break Warnings**
   - Use `torch._dynamo.explain()` to identify causes
   - Refactor code to avoid problematic patterns
   - Add compilation guards where necessary

3. **Triton Compilation Errors**
   - Check CUDA compatibility and versions
   - Verify kernel constraints and type annotations
   - Use proper masking for boundary conditions

4. **Memory Issues**
   - Reduce batch sizes for large models
   - Enable gradient checkpointing
   - Monitor GPU memory usage with `nvidia-smi`

### Debug Commands

```bash
# Check PyTorch compilation status
export TORCH_COMPILE_DEBUG=1

# Enable verbose Triton output
export TRITON_PRINT_AUTOTUNING=1

# Debug CUDA kernel launches
export CUDA_LAUNCH_BLOCKING=1
```

## Advanced Features

### Custom Autotuning

```python
# Triton autotuning for optimal configuration
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def autotuned_kernel(...):
    pass
```

### Advanced torch.compile Features

```python
# Custom backend integration
@torch.compile(backend="custom_backend")
def model(x):
    return x

# Selective compilation
@torch.compile(disable_dynamic_shapes=True)
def static_model(x):
    return x
```

## Best Practices

1. **Start Simple**: Begin with basic compilation before advanced optimizations
2. **Profile First**: Measure before and after optimization
3. **Iterative Optimization**: Apply optimizations incrementally
4. **Hardware Awareness**: Consider target GPU architecture
5. **Testing**: Verify numerical correctness after optimization

For more advanced techniques and detailed explanations, refer to Chapter 14 of the AI Performance Engineering book and the official PyTorch and Triton documentation.
