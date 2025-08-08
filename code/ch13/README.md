# Chapter 13: Profiling, Tuning, and Scaling PyTorch

This directory contains comprehensive examples for profiling, debugging, and optimizing PyTorch workloads across single-GPU and distributed multi-GPU environments.

## Files Overview

### Core Profiling Examples

- **`train_deepseek_v3.py`** - Complete profiling example with PyTorch Profiler, NVTX markers, and trace export
- **`memory_profiling.py`** - Comprehensive memory profiling and optimization techniques
- **`fsdp_example.py`** - Fully Sharded Data Parallel (FSDP) implementation with memory efficiency
- **`custom_allocator.py`** - Custom CUDA allocator configuration and memory pool management

## Key Profiling Tools Covered

### PyTorch Profiler (Kineto)
- Operator-level CPU/GPU profiling
- Memory usage tracking
- NVTX marker integration
- Chrome trace export for visualization

### NVIDIA Nsight Tools
- **Nsight Systems (nsys)**: System-wide timeline profiling
- **Nsight Compute (ncu)**: Detailed GPU kernel analysis
- Multi-process and multi-node support

### Memory Analysis
- PyTorch memory profiler
- Memory snapshots and visualization
- Custom allocator configuration
- Distributed memory management

### Holistic Trace Analysis (HTA)
- Multi-GPU trace visualization
- Communication/computation overlap analysis
- Performance bottleneck identification

## Usage Examples

### Basic Profiling with PyTorch Profiler

```bash
# Run training with profiling enabled
python train_deepseek_v3.py

# View generated trace in Chrome
# Open chrome://tracing and load deepseek_v3_trace.json
```

### Memory Profiling and Optimization

```bash
# Comprehensive memory analysis
python memory_profiling.py

# Monitor memory usage during training
python -m memory_profiler train_deepseek_v3.py
```

### FSDP Distributed Training

```bash
# Single-node FSDP training
python fsdp_example.py

# Multi-node FSDP training
torchrun --nproc_per_node=4 --nnodes=2 fsdp_example.py
```

### Custom Memory Allocator

```bash
# Test different allocator configurations
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python custom_allocator.py

# Enable memory history for debugging
python custom_allocator.py
```

## Advanced Profiling Workflows

### 1. Holistic Performance Analysis

```bash
# Step 1: Generate PyTorch Profiler traces
python train_deepseek_v3.py

# Step 2: Analyze with HTA (requires installation)
# hta trace_analyzer --trace_dir ./hta_traces

# Step 3: System-wide profiling with Nsight Systems
# nsys profile --trace=cuda,nvtx python train_deepseek_v3.py

# Step 4: Kernel-level analysis with Nsight Compute
# ncu --set full python train_deepseek_v3.py
```

### 2. Multi-GPU Performance Debugging

```bash
# Profile distributed training
torchrun --nproc_per_node=2 train_deepseek_v3.py

# Generate per-rank traces for HTA
TORCH_TRACE_DIR=./traces torchrun --nproc_per_node=2 train_deepseek_v3.py

# Analyze communication patterns
# hta trace_analyzer --trace_dir ./traces
```

### 3. Memory Optimization Pipeline

```python
# 1. Profile baseline memory usage
torch.cuda.memory._record_memory_history(True)
# ... run training ...
snapshot = torch.cuda.memory._snapshot()

# 2. Apply optimizations
# - Gradient checkpointing
# - Mixed precision training  
# - FSDP parameter sharding

# 3. Measure improvements
# - Compare peak memory usage
# - Analyze fragmentation patterns
# - Verify correctness
```

## Performance Optimization Techniques

### Memory Optimizations

1. **Gradient Checkpointing**
   ```python
   x = torch.utils.checkpoint.checkpoint(layer, x)
   ```

2. **Mixed Precision Training**
   ```python
   with torch.cuda.amp.autocast():
       output = model(input)
   ```

3. **Memory-Efficient Attention**
   ```python
   torch.nn.functional.scaled_dot_product_attention(q, k, v)
   ```

4. **FSDP Parameter Sharding**
   ```python
   model = FSDP(model, auto_wrap_policy=transformer_auto_wrap_policy)
   ```

### Compute Optimizations

1. **torch.compile() for Kernel Fusion**
   ```python
   compiled_model = torch.compile(model, mode="max-autotune")
   ```

2. **CUDA Graphs for Reduced Overhead**
   ```python
   graph = torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):
       output = model(static_input)
   ```

3. **Efficient Data Loading**
   ```python
   dataloader = DataLoader(dataset, pin_memory=True, num_workers=4)
   ```

## Profiler Configuration Examples

### PyTorch Profiler Setup

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    use_cuda=True
) as prof:
    # Training code here
    pass

# Export traces
prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### NVTX Marker Integration

```python
import torch.profiler

with torch.profiler.record_function("training_step"):
    # Forward pass
    with torch.profiler.record_function("forward"):
        output = model(input)
    
    # Backward pass  
    with torch.profiler.record_function("backward"):
        loss.backward()
```

### Memory Allocator Configuration

```bash
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6"

# Enable memory debugging
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## Integration with External Tools

### Nsight Systems Integration

```bash
# Profile with detailed CUDA info
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output=nsys_trace \
    python train_deepseek_v3.py

# View results
nsys-ui nsys_trace.nsys-rep
```

### HTA Analysis

```bash
# Install HTA
pip install git+https://github.com/facebookresearch/HolisticTraceAnalysis.git

# Generate traces with proper naming
TORCH_TRACE_DIR=./hta_traces python train_deepseek_v3.py

# Analyze with HTA
hta trace_analyzer --trace_dir ./hta_traces
```

### TorchBench Integration

```bash
# Install TorchBench
git clone https://github.com/pytorch/benchmark.git
cd benchmark && python install.py

# Run standardized benchmarks
python run_benchmark.py --model_name transformer --device cuda
```

## Performance Metrics and KPIs

### Key Metrics to Track

1. **Throughput**: Tokens/second, samples/second
2. **Memory Efficiency**: Peak usage, fragmentation ratio
3. **GPU Utilization**: SM occupancy, memory bandwidth utilization
4. **Communication Efficiency**: All-reduce latency, overlap percentage
5. **End-to-End Latency**: Time per training step

### Automated Performance Monitoring

```python
# Example performance tracking
class PerformanceTracker:
    def __init__(self):
        self.start_time = None
        self.samples_processed = 0
        
    def start_step(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        
    def end_step(self, batch_size):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        self.samples_processed += batch_size
        
        throughput = batch_size / elapsed
        memory_used = torch.cuda.memory_allocated() / 1e9
        
        return {
            'throughput': throughput,
            'memory_gb': memory_used,
            'step_time_ms': elapsed * 1000
        }
```

## Troubleshooting Common Issues

### Performance Bottlenecks

1. **Low GPU Utilization**
   - Check data loading pipeline
   - Verify batch size is optimal
   - Look for CPU-GPU synchronization points

2. **High Memory Usage**
   - Enable gradient checkpointing
   - Use FSDP for parameter sharding
   - Implement mixed precision training

3. **Slow Communication**
   - Verify NCCL backend is used
   - Check network topology
   - Enable communication/computation overlap

### Debugging Tools

```bash
# Enable debug logging
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Memory debugging
export PYTORCH_CUDA_ALLOC_CONF="backend:native,debug:True"

# Compilation debugging  
export TORCH_COMPILE_DEBUG=1
```

## Best Practices

1. **Profile Early and Often**: Integrate profiling into development workflow
2. **Focus on Bottlenecks**: Use 80/20 rule - optimize the slowest components first
3. **Measure Impact**: Always benchmark before and after optimizations
4. **Monitor in Production**: Set up continuous performance monitoring
5. **Share Results**: Use trace exports for collaboration and debugging

For more advanced techniques, refer to the PyTorch documentation and NVIDIA developer resources.
