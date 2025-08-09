# Chapter 19: Adaptive Parallelism Strategy and Token-Level Precision Switching

## Summary
These examples demonstrate adaptive parallelism and token‑level precision switching (with dynamic KV‑cache quantization) to improve throughput and tail latency.

## Performance Takeaways
- Select TP/PP/DP/Hybrid strategies at runtime based on workload signals
- Use cooldown and hysteresis to avoid strategy‑switch thrashing
- Adapt token precision by confidence to save compute without hurting quality
- Quantize KV cache dynamically to relieve memory pressure and boost capacity
- Reduce tail latency and raise throughput across mixed request patterns

Code examples demonstrating adaptive parallelism strategies and dynamic precision switching for efficient inference optimization.

## Examples

- `dynamic_parallelism.py` - Adaptive parallelism strategy selection based on workload
- `token_precision_switch.py` - Token-level precision switching during generation

## Key Concepts

- **Dynamic Parallelism**: Runtime adaptation between tensor, pipeline, and hybrid parallelism
- **Token-Level Precision**: Adaptive precision switching based on model confidence 
- **Workload-Aware Routing**: Intelligent strategy selection for different request patterns
- **Confidence Metrics**: Entropy, logit variance, and probability-based quality assessment
- **Dynamic Quantization**: Real-time KV cache compression based on memory pressure

## Requirements

- CUDA 12.8+
- PyTorch 2.8+ with distributed support
- HQQ library (optional, for advanced quantization)
- GPU with compute capability 7.0+ for optimal mixed precision

## Usage

### Dynamic Parallelism Demo
```bash
# Run adaptive parallelism simulation
python dynamic_parallelism.py

# Demonstrates:
# - Automatic strategy switching based on workload
# - Performance evaluation across different scenarios
# - Cooldown mechanisms to prevent thrashing
```

### Token Precision Switching Demo  
```bash
# Run token-level precision adaptation
python token_precision_switch.py

# Shows:
# - Confidence-based precision switching
# - Dynamic KV cache quantization
# - Performance overhead analysis
```

## Parallelism Strategies

The dynamic parallelism router chooses between:

### Tensor Parallel (TP)
- **Best for**: Latency-sensitive, moderate sequences (< 1024 tokens)
- **Configuration**: TP=8, PP=1, DP=1  
- **Latency**: ~50ms, high bandwidth utilization

### Pipeline Parallel (PP)
- **Best for**: Very long sequences (> 2048 tokens), memory-constrained
- **Configuration**: TP=1, PP=8, DP=1
- **Memory efficiency**: 90%, handles memory pressure well

### Hybrid (TP+PP)
- **Best for**: Long sequences with moderate memory pressure
- **Configuration**: TP=4, PP=2, DP=1
- **Balanced**: Good latency and memory efficiency

### Data Parallel (DP)
- **Best for**: High throughput, many short sequences (< 256 tokens)
- **Configuration**: TP=1, PP=1, DP=8
- **Throughput**: ~1500 tokens/sec across batch

## Decision Algorithm

From Chapter 19's routing logic:
```python
def choose_worker_pool(seq_len, gpu_mem_util, concurrent_reqs):
    # Long contexts or high memory pressure -> hybrid/pipeline
    if seq_len > 1024 or gpu_mem_util > 0.8:
        return HYBRID if gpu_mem_util < 0.9 else PIPELINE_PARALLEL
    
    # Many concurrent short requests -> data parallel  
    elif concurrent_reqs > 32 and seq_len < 256:
        return DATA_PARALLEL
        
    # Latency-sensitive -> tensor parallel
    else:
        return TENSOR_PARALLEL
```

## Precision Switching

Token-level precision adapts based on confidence metrics:

### Confidence Assessment
- **High confidence** (score > 0.9): Switch to INT8/INT4
- **Medium confidence** (0.6-0.9): Use FP16
- **Low confidence** (< 0.6): Use FP32 for stability

### Confidence Metrics
```python
confidence_score = (
    0.5 * max_probability +      # Softmax confidence
    0.3 * (1 - entropy/4.0) +    # Uncertainty measure  
    0.2 * logit_diff/10.0        # Top-2 separation
)
```

### Dynamic Quantization
- Monitors GPU memory usage in real-time
- Switches between 16-bit, 8-bit, and 4-bit KV cache
- Uses async quantization to minimize compute impact

## Performance Benefits

### Dynamic Parallelism
- **2-4× better resource utilization** across mixed workloads
- **Automatic adaptation** to traffic patterns
- **Reduced tail latency** through optimal strategy selection

### Token Precision Switching  
- **15-30% speedup** for high-confidence tokens
- **Minimal quality loss** (~1% BLEU score impact)
- **Memory savings** up to 50% with dynamic quantization

## Monitoring and Tuning

Key metrics to track:
- Strategy switch frequency (avoid thrashing)
- Precision usage distribution 
- Confidence score accuracy
- Memory pressure trends
- Latency/throughput per strategy

Configure cooldown periods and thresholds based on:
- Model size and architecture
- Hardware capabilities (memory, compute)
- SLA requirements (latency vs throughput)
- Quality tolerance for your application
