# Performance Test Harness Guide

## Overview
This document describes the timing mechanisms and best practices used in the AI Performance Engineering codebase for CUDA 13, PyTorch 2.9, and Triton 3.5.0.

## Timing Best Practices

### ✅ GPU Timing (CUDA Events) - PREFERRED
Use CUDA Events for accurate GPU-side timing:

```python
# Create events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Warmup
for _ in range(10):
    model(data)

# Time execution
torch.cuda.synchronize()
start.record()

for _ in range(iters):
    output = model(data)

end.record()
end.synchronize()

# Get elapsed time in milliseconds
elapsed_ms = start.elapsed_time(end)
avg_ms_per_iter = elapsed_ms / iters
```

**Why CUDA Events?**
- Measures GPU execution time (not wall-clock time)
- Accounts for asynchronous kernel launches
- More accurate than `time.time()` for GPU workloads
- Built-in support for multi-stream timing

### ❌ CPU Timing - AVOID for GPU code
```python
# DON'T DO THIS for GPU timing
start = time.time()
output = model(data)  # Async launch returns immediately!
elapsed = time.time() - start  # Measures host time, not GPU time
```

**When CPU timing is acceptable:**
- Pure CPU workloads
- Distributed operations that involve network latency
- End-to-end system timing including I/O

## Test Harness Specifications

### ch4/symmetric_memory_example.py
- **Timing**: ✅ CUDA Events
- **Warmup**: 10 iterations
- **Measurement**: 100 iterations
- **Timeout**: 30 seconds (distributed timeout)
- **Expected Runtime**: 5-10 seconds on 2 GPUs
- **Sizes Tested**: 4 KB, 1 MB, 4 MB

### ch7/vectorized_copy.cu
- **Timing**: ✅ CUDA Events (`cudaEventElapsedTime`)
- **Warmup**: Implicit (first kernel launch)
- **Size**: 1M floats (4 MB)
- **Expected Runtime**: <1 second
- **Tests**: 16-byte vs 32-byte alignment

### ch14/torch_compiler_examples.py
- **Timing**: ✅ CUDA Events (FIXED in this update)
- **Warmup**: 3 iterations
- **Measurement**: 10 iterations per mode
- **Expected Runtime**: 10-20 seconds total (3 modes)
- **Tests**: default, reduce-overhead, max-autotune modes

### ch14/triton_nvshmem_example.py
- **Timing**: N/A (educational/demo only)
- **Expected Runtime**: 2-5 seconds
- **Purpose**: Demonstrates API usage, not performance measurement

## Iteration Count Guidelines

| Test Type | Warmup | Measurement | Rationale |
|-----------|--------|-------------|-----------|
| Micro-benchmark (kernel) | 5-10 | 20-100 | Fast execution, need more samples |
| Model inference | 3-5 | 10-20 | Slower, fewer samples needed |
| Training step | 2-3 | 5-10 | Slowest, minimal samples |
| Distributed operation | 3-5 | 10-50 | Network variance, more samples |

## Timeout Mechanisms

### Distributed Operations
Always set timeouts for distributed initialization:

```python
dist.init_process_group(
    backend="nccl",
    timeout=torch.distributed.timedelta(seconds=30)
)
```

### Long-Running Tests
Add explicit timeout checks:

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test exceeded maximum runtime")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute timeout

try:
    run_test()
finally:
    signal.alarm(0)  # Cancel alarm
```

## CUDA Synchronization

### When to Synchronize

1. **Before timing**: Ensure all prior work completes
   ```python
   torch.cuda.synchronize()
   start.record()
   ```

2. **After timing**: Wait for timed work to complete
   ```python
   end.record()
   end.synchronize()  # or torch.cuda.synchronize()
   ```

3. **Between dependent operations**: Only if needed
   ```python
   # Usually NOT needed - streams handle this
   kernel1<<<...>>>()
   # torch.cuda.synchronize()  # Unnecessary!
   kernel2<<<...>>>()  # Automatically waits if using same stream
   ```

### When NOT to Synchronize

- Between operations on the same stream (automatic ordering)
- Inside tight loops (kills parallelism)
- After every kernel launch (massive overhead)

## Memory Size Considerations

Choose test sizes that:
1. **Complete quickly**: < 100ms per iteration for interactive tests
2. **Exceed cache**: Large enough to measure memory bandwidth
3. **Fit in memory**: Leave headroom for multiple buffers
4. **Align with hardware**: Multiples of 256 bytes for coalescing

### Example Sizes

| Purpose | Size | Rationale |
|---------|------|-----------|
| L1/L2 cache test | 1-10 KB | Fits in cache |
| Memory bandwidth | 10-100 MB | Measures DRAM |
| Large model inference | 100 MB - 1 GB | Realistic workload |
| Stress test | > 1 GB | Test scaling |

## NVTX Profiling Integration

Add NVTX ranges for Nsight profiling:

```python
import torch.cuda.nvtx as nvtx

with nvtx.range("model_forward"):
    output = model(input)

with nvtx.range("loss_backward"):
    loss.backward()
```

**Benefits:**
- Visual markers in Nsight Systems timeline
- Easy identification of bottlenecks
- Minimal overhead when not profiling

## Common Pitfalls

### ❌ Pitfall 1: No Warmup
```python
# Cold start includes compilation/caching overhead
start.record()
output = model(data)  # First run is always slower!
end.record()
```

**Fix**: Always warmup
```python
for _ in range(3):
    model(data)  # Warmup
torch.cuda.synchronize()
start.record()
# ... actual timing ...
```

### ❌ Pitfall 2: Measuring Async Operations
```python
start.record()
dist.send(tensor)  # Returns immediately
end.record()  # Doesn't measure actual network time!
```

**Fix**: Use barriers or wait for completion
```python
start.record()
dist.send(tensor)
dist.barrier()  # Wait for completion
end.record()
```

### ❌ Pitfall 3: Insufficient Iterations
```python
# 1 iteration - high variance!
elapsed = time_once()
```

**Fix**: Multiple iterations with statistics
```python
times = [time_once() for _ in range(20)]
avg = np.mean(times)
std = np.std(times)
print(f"Time: {avg:.2f} ± {std:.2f} ms")
```

## Validation Checklist

Before committing a test:

- [ ] Uses CUDA Events for GPU timing (not time.time)
- [ ] Includes warmup iterations (3-10)
- [ ] Measures multiple iterations (10-100)
- [ ] Has explicit timeout for distributed ops
- [ ] Documents expected runtime in docstring
- [ ] Completes in < 60 seconds for CI
- [ ] Includes torch.cuda.synchronize() before/after timing
- [ ] Reports average time per iteration
- [ ] Handles CPU-only fallback gracefully
- [ ] Includes NVTX ranges for profiling

## Summary: Our Updated Tests

| File | Timing Mechanism | Warmup | Iters | Timeout | Runtime | Status |
|------|-----------------|--------|-------|---------|---------|--------|
| `symmetric_memory_example.py` | ✅ CUDA Events | 10 | 100 | 30s | 5-10s | ✅ GOOD |
| `vectorized_copy.cu` | ✅ CUDA Events | Implicit | 1 | N/A | <1s | ✅ GOOD |
| `torch_compiler_examples.py` | ✅ CUDA Events | 3 | 10 | N/A | 10-20s | ✅ FIXED |
| `triton_nvshmem_example.py` | N/A (demo) | N/A | N/A | N/A | 2-5s | ✅ GOOD |

All tests now follow best practices for GPU performance measurement!
