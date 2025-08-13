# Chapter 13: Profiling, Tuning, and Scaling PyTorch

This chapter demonstrates comprehensive profiling, debugging, and system-level tuning of PyTorch workloads running on modern NVIDIA GPUs. The examples cover identifying and fixing bottlenecks using PyTorch's built-in profiler, NVIDIA's Nsight tools, CPU profiling with Linux perf, PyTorch memory profiling, and memory allocator tuning.

## Overview

The chapter focuses on:
- **Holistic Profiling**: Using multiple tools to profile across the entire system stack
- **PyTorch Compiler**: Leveraging torch.compile for automatic optimizations
- **CUDA Streams**: Overlapping computation and communication
- **CUDA Graphs**: Reducing kernel launch overhead
- **Memory Optimization**: Tuning allocators and using checkpointing
- **Distributed Training**: FSDP, DDP, and multi-GPU scaling
- **Performance Monitoring**: Continuous integration and benchmarking

## Files

### Main Training Example
- `train_deepseek_v3.py` - Comprehensive DeepSeek-V3 training example with all profiling and optimization techniques

### Profiling Examples
- `profiling_example/compiler_comparison.py` - PyTorch compiler mode comparison and debugging
- `profiling_example/train_deepseek_v3.py` - Focused DeepSeek training with profiling techniques

## Key Techniques Demonstrated

### 1. PyTorch Profiler with NVTX Markers

```python
with profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
) as prof:
    with profiler.record_function("train_step"):
        torch.cuda.nvtx.range_push("forward")
        # ... forward pass
        torch.cuda.nvtx.range_pop()
```

### 2. PyTorch Compiler (torch.compile)

```python
# Different compilation modes
compiled_model = torch.compile(model, mode="max-autotune")
compiled_model = torch.compile(model, mode="reduce-overhead")
compiled_model = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

### 3. CUDA Streams for Overlapping

```python
transfer_stream = torch.cuda.Stream(device)
compute_stream = torch.cuda.default_stream(device)

# Overlap data transfer with computation
with torch.cuda.stream(transfer_stream):
    next_inputs = batch[0].to(device, non_blocking=True)

compute_stream.wait_stream(transfer_stream)
# Use the transferred data
```

### 4. CUDA Graphs

```python
g = torch.cuda.CUDAGraph()
capture_stream = torch.cuda.Stream()

# Warm up
with torch.cuda.stream(capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)

# Capture graph
with torch.cuda.graph(g, stream=capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)

# Replay graph
g.replay()
```

### 5. FSDP with Automatic Checkpointing

```python
fsdp_model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(
        offload_params=True,
        offload_gradients=True),
    activation_checkpointing_policy={
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.MultiheadAttention,
    }
)
```

### 6. Memory Allocator Configuration

```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:128,'
    'roundup_power2_divisions:[256:1,512:2,1024:4,>:8],'
    'garbage_collection_threshold:0.8,'
    'backend:cudaMallocAsync'
)
```

## Profiling Tools Covered

### PyTorch Profiler (Kineto)
- In-PyTorch op-level profiling (CPU/GPU)
- NVTX marker support, shape recording, memory stats
- Identifies compile graph breaks

### Nsight Systems (nsys)
- System-wide timeline (CPU, GPU, OS, I/O)
- Unified timeline of CPU threads & GPU streams
- Multi-process support

### Nsight Compute (ncu)
- GPU kernel analysis (per-kernel)
- Per-kernel hardware metrics, source correlation
- Roofline analysis, occupancy & throughput reports

### PyTorch Memory Profiler
- GPU memory usage by operation
- Memory snapshot timeline, per-op peak memory
- Integration with torch.cuda.memory_stats()

### Linux perf
- CPU profiling & system events
- Sampling of CPU cycles/instructions/cache
- Flame graphs, off-CPU analysis

### Holistic Trace Analysis (HTA)
- Distributed training trace visualization
- Browser-based Kineto trace explorer
- Multi-worker trace aggregation

## Compilation Modes

| Mode | Description | Compile Time | Extra Memory | Features |
|------|-------------|--------------|--------------|----------|
| default | Balanced optimizations | Low-Medium | No | General fusion, basic autotuning |
| reduce-overhead | Reduces per-iteration overhead | Medium | Yes | Uses CUDA Graphs (if possible) |
| max-autotune | Maximizes runtime performance | High | Maybe | Aggressive Triton autotuning |
| max-autotune-no-cudagraphs | Same as max-autotune but without graphs | High | No | Same as above but disables graphs |

## Attention Optimization Techniques

1. **Scaled Dot Product Attention (SPDA)**
   - Use `torch.nn.functional.scaled_dot_product_attention`
   - Automatically uses fastest available kernel (e.g. FlashAttention)

2. **FlexAttention**
   - Compiler-based approach for custom sparsity patterns
   - Can be 2x faster for specific sparse attention patterns

3. **FlexDecoding**
   - Optimizes decoder side of sequence generation
   - Uses KV caching efficiently across timesteps

4. **Context Parallel**
   - Parallelizes attention across multiple implementations
   - Uses `torch.context_parallel()` context manager

## Memory Optimization Techniques

### Activation Checkpointing
```python
from torch.utils.checkpoint import checkpoint

# Wrap model layers to save memory
output = checkpoint(model_layer, input)
```

### Memory Allocator Tuning
- Configure `PYTORCH_CUDA_ALLOC_CONF` environment variable
- Use `max_split_size_mb` to reduce fragmentation
- Enable `cudaMallocAsync` backend for better performance

### Offloading
- CPU offloading for parameters and gradients
- NVMe offloading for very large models
- Overlap transfers with computation

## Distributed Training

### DDP with torch.compile
- Automatic graph breaks at synchronization points
- Overlaps communication with computation
- Tune bucket size for optimal performance

### FSDP with torch.compile
- Wrap at Transformer-block granularity
- Use `use_orig_params=True` for better compatibility
- Enable automatic checkpointing and offloading

## Performance Monitoring

### Continuous Integration
```yaml
- name: Run DeepSeek MoE benchmark
  run: |
    torchbench run --model deepseek_moe --iters 10 --batch-size 4 --json results.json
- name: Compare throughput
  run: |
    python scripts/compare_perf.py baseline.json results.json
```

### MLPerf Logging
```python
log_entry = {
    "step_time_ms": 24.0,
    "forward_ms": 10.5,
    "backward_ms": 9.0,
    "allreduce_ms": 4.0,
    "other_ms": 0.5
}
print(f":::MLL {log_entry}")
```

## Key Takeaways

1. **Use a profile-first approach** - Don't optimize based on intuition alone
2. **Enable torch.compile** - Easy speedups with minimal code changes
3. **Use the highest optimization mode** - max-autotune for long-running jobs
4. **Avoid synchronization gotchas** - Use non-blocking transfers and stream events
5. **Utilize Tensor Cores** - Enable mixed precision for better performance
6. **Fuse small operations** - Use torch.compile or custom fused kernels
7. **Reduce memory fragmentation** - Configure allocator and reuse buffers
8. **Use activation checkpointing** - Trade compute for memory
9. **Offload memory** - Use CPU and NVMe for large models
10. **Optimize data pipeline** - Use multiple workers and prefetching
11. **Monitor performance continuously** - Set up CI and benchmarking

## Running the Examples

### Prerequisites
- PyTorch 2.8+
- CUDA 12.9+
- Transformers library
- NVIDIA GPU with sufficient memory

### Basic Usage
```bash
# Run the main training example
python train_deepseek_v3.py

# Run compiler comparison
python profiling_example/compiler_comparison.py

# Run focused DeepSeek profiling
python profiling_example/train_deepseek_v3.py
```

### Environment Variables
```bash
# Enable compiler logging
export TORCH_LOGS="+dynamo,+inductor"

# Configure memory allocator
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,backend:cudaMallocAsync"

# Enable unique kernel names for benchmarking
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_BENCHMARK_KERNEL=1
```

## Performance Tips

1. **Always profile first** - Use PyTorch Profiler to identify bottlenecks
2. **Start with torch.compile** - Often provides significant speedups with minimal effort
3. **Monitor memory usage** - Use memory profiler to identify allocation hotspots
4. **Use appropriate compilation modes** - Choose based on your workload characteristics
5. **Overlap computation and communication** - Use CUDA streams effectively
6. **Optimize data loading** - Use multiple workers and prefetching
7. **Set up continuous monitoring** - Catch performance regressions early
8. **Test with real workloads** - Benchmarks may not reflect production performance

## Troubleshooting

### Common Issues

1. **Graph breaks in torch.compile**
   - Use `torch._dynamo.explain()` to identify causes
   - Check for dynamic control flow or unsupported operations

2. **Memory fragmentation**
   - Configure memory allocator with `PYTORCH_CUDA_ALLOC_CONF`
   - Use fixed-size buffers when possible

3. **Poor GPU utilization**
   - Check for data loading bottlenecks
   - Use Nsight Systems to identify gaps in GPU timeline

4. **Slow distributed training**
   - Tune bucket sizes for DDP
   - Use appropriate sharding strategy for FSDP
   - Monitor network bandwidth utilization

### Debugging Commands

```bash
# Profile with Nsight Systems and export stats
nsys profile -o profile python train_deepseek_v3.py
nsys stats --report summary,cuda_api --format sqlite,csv profile -o profile

# Profile with Nsight Compute
ncu --kernel-name-regex "matmul" python train_deepseek_v3.py

# CPU profiling with perf
perf record -F 2000 -g python train_deepseek_v3.py
perf report

# Memory profiling
python -m torch.utils.memory_viz train_deepseek_v3.py
```

This chapter provides a comprehensive toolkit for profiling, optimizing, and scaling PyTorch workloads. The examples demonstrate both the tools and techniques needed to achieve maximum performance on modern GPU hardware.
