# Chapter 17: Scaling Disaggregated Prefill and Decode for Inference

This chapter covers disaggregated prefill and decode architectures, routing policies, scheduling algorithms, and quality-of-service mechanisms for large-scale LLM inference systems.

## Key Topics Covered

### Why Prefill-Decode Disaggregation?
- **Interference Elimination**: Separating prefill and decode phases to remove head-of-line blocking
- **Phase-Specific Optimizations**: Independent tuning of compute-bound prefill and memory-bound decode
- **Latency Improvements**: Achieving <200ms TTFT and <50ms TPOT simultaneously
- **Throughput Optimization**: Higher goodput under latency constraints

### Disaggregated Prefill-Decode Cluster Pools
- **Prefill Workers**: Compute-optimized nodes for prompt processing
- **Decode Workers**: Memory-optimized nodes for token generation
- **KV Cache Transfer**: High-speed interconnects and RDMA transfers
- **Dynamic Scaling**: Independent scaling of prefill and decode pools

### Disaggregated Routing and Scheduling Policies
- **Routing Factors**: Prompt length, cache hits, queue lengths, worker load
- **Dynamic Routing**: Conditional disaggregation based on system state
- **Latency-Aware Routing**: Multi-factor scoring for optimal worker selection
- **Capacity-Aware Routing**: Real-time adaptation to workload changes

### Quality-of-Service (QoS) and Early Rejection
- **Admission Control**: Early rejection to maintain SLOs under load
- **Priority-Based Scheduling**: Different service levels for different request types
- **Graceful Degradation**: Service quality reduction instead of complete failure
- **SLO Tracking**: Real-time monitoring of TTFT and TPOT targets

## Code Examples

### Dynamo Configuration Example

The chapter includes a YAML configuration example for NVIDIA Dynamo:

```yaml
{
  "model": "llama-70b",
  "split_policy": {
    "prompt_length_threshold": 256,
    "prefix_cache_weight": 10.0,
    "queue_length_weight": 1.5,
    "decode_load_weight": 0.5,
    "enable_hotspot_prevention": true
  },
  "cache": {
    "reuse_prefix": true,
    "min_cache_hit_ratio": 0.75
  },
  "autoscale": {
    "prefill": {
      "min_replicas": 4,
      "max_replicas": 12,
      "scale_up": { "queue_length": 8, "gpu_utilization": 80 },
      "scale_down": { "queue_length": 2, "gpu_utilization": 40 }
    },
    "decode": {
      "min_replicas": 8,
      "max_replicas": 24,
      "scale_up": { "queue_length": 16, "kv_cache_usage": 75 },
      "scale_down": { "queue_length": 4, "kv_cache_usage": 30 }
    }
  },
  "qos": {
    "enable_early_rejection": true,
    "low_priority_threshold_ms": 500,
    "reject_on_slo_violation": true
  }
}
```

### Dynamic Routing Policy Example

The chapter includes a Python example showing dynamic routing logic:

```python
# Offload prefill decision running on a decode worker
def should_offload_prefill(prompt_length, prefix_cached_length, prefill_queue_size):
    # Condition 1: Long effective prefill
    # (prompt minus cached part exceeds threshold)
    long_prefill = (prompt_length - prefix_cached_length) > PREFILL_LENGTH_THRESHOLD
    
    # Condition 2: Prefill workers have capacity
    # (queue not too long)
    prefill_available = prefill_queue_size < PREFILL_QUEUE_MAX
    
    if long_prefill and prefill_available:
        # offload to prefill worker
        return True
    else:
        # do prefill locally (async)
        return False
```

### Early Rejection Policy Example

The chapter includes a Python example showing early rejection logic:

```python
# Early rejection based on estimated latency and priority
def admit_request(priority):
    # Estimate current TTFT when new request is added
    est_ttft = get_current_prefill_queue_length() * get_avg_prefill_time_per_req()
    
    # Consider decode backlog as well
    est_ttft += get_current_decode_queue_length() * get_avg_decode_time_per_req()
    
    if est_ttft > TTFT_SLO_MAX:
        if priority == "low":
            # reject low priority request
            return False
        else:
            # high priority: admit high priority request
            return True
    else:
        return True
```

## Architecture Considerations

### Hardware Requirements
- **NVIDIA Blackwell B200/B300**: SM100 compute capability
- **CUDA 12.9**: Latest CUDA toolkit with advanced features
- **PyTorch 2.8**: Latest PyTorch with optimized inference
- **High-bandwidth Interconnects**: NVLink, NVSwitch, InfiniBand
- **Multi-node Clusters**: Distributed inference across multiple nodes

### Performance Targets
- **TTFT (Time-to-First-Token)**: <200ms p99 for interactive applications
- **TPOT (Time-Per-Output-Token)**: <50ms p99 for streaming generation
- **Throughput**: >1000 requests/second for large models
- **Cache Hit Rate**: >80% for efficient routing

## Profiling and Monitoring

### Nsight Systems (nsys)
```bash
# Profile disaggregated routing
nsys profile -t cuda,osrt -o routing_profile python dynamic_routing.py
nsys profile -t cuda,osrt -o rejection_profile python early_rejection.py
nsys profile -t cuda,osrt -o latency_profile python latency_aware_routing.py

# Profile KV cache transfer
nsys profile -t cuda,osrt -o kv_transfer_profile python kv_cache_transfer.py
nsys profile -t cuda,osrt -o nixl_profile python nixl_transfer.py
```

### Nsight Compute (ncu)
```bash
# Profile kernel efficiency
ncu --metrics achieved_occupancy,warp_execution_efficiency -o routing_profile python dynamic_routing.py
ncu --metrics dram_read_throughput,dram_write_throughput -o memory_profile python kv_cache_transfer.py

# Profile with PC sampling
ncu --sampling-period=1000 -o pc_sampling_profile python latency_aware_routing.py
```

### PyTorch Profiler
```bash
# Profile disaggregated inference
python -m torch.utils.bottleneck dynamic_routing.py

# Profile with memory tracking
python -c "
import torch
import torch.profiler as profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    import dynamic_routing
    dynamic_routing.main()

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
"
```

## Expected Performance Characteristics

### Disaggregated Architecture
- **Interference Reduction**: <5% cross-stage interference
- **Latency Improvement**: 2-3x better TTFT and TPOT
- **Throughput Enhancement**: 30-50% higher goodput
- **Resource Utilization**: Near-100% GPU utilization per phase

### Routing Performance
- **Routing Decisions**: 1000+ requests/second
- **Offload Rate**: 60-80% for long prompts
- **Cache Hit Rate**: 80%+ for repeated patterns
- **Routing Accuracy**: 90%+ optimal worker selection

### QoS and Early Rejection
- **Rejection Rate**: <10% under normal load
- **SLO Compliance**: >95% for high-priority requests
- **Graceful Degradation**: Service quality reduction instead of failure
- **Priority Handling**: 3-5x faster for high-priority requests

### KV Cache Transfer
- **Transfer Size**: 1-5 GB per request
- **Transfer Time**: 5-15ms with RDMA
- **Bandwidth Utilization**: 90%+ with NIXL
- **CPU Overhead**: <1% with GPU-direct transfers

## Implementation Guidelines

### Disaggregated Setup
1. Configure separate prefill and decode worker pools
2. Set up high-speed interconnects for KV cache transfer
3. Implement dynamic routing based on prompt length and cache hits
4. Monitor TTFT and TPOT metrics continuously

### Routing Configuration
1. Set prompt length thresholds for offload decisions
2. Configure cache-aware routing for optimal worker selection
3. Implement adaptive routing based on queue lengths
4. Use latency-aware scoring for worker selection

### QoS Implementation
1. Set up early rejection thresholds based on SLOs
2. Implement priority-based scheduling for different request types
3. Configure graceful degradation for overload scenarios
4. Monitor and alert on SLO violations

### Scaling Strategy
1. Use independent scaling for prefill and decode pools
2. Implement adaptive scaling based on queue lengths
3. Monitor resource utilization and adjust accordingly
4. Use heterogeneous hardware for cost optimization

## Troubleshooting

### Common Issues
- **High TTFT**: Optimize prefill workers and reduce queue delays
- **High TPOT**: Optimize decode workers and continuous batching
- **Memory Pressure**: Implement KV cache offloading and memory pooling
- **Load Imbalance**: Use adaptive routing and dynamic scaling
- **SLO Violations**: Implement early rejection and QoS policies

### Performance Tuning
1. Profile routing decisions and optimize thresholds
2. Monitor cache hit rates and adjust routing policies
3. Track queue lengths and scale workers accordingly
4. Implement adaptive scaling based on workload patterns
5. Use heterogeneous hardware for cost optimization

## Key Takeaways

1. **Eliminate Prefill-Decode Interference**: Separate phases to remove head-of-line blocking
2. **Optimize Each Phase Independently**: Use different hardware and parallelism for prefill vs decode
3. **Leverage the KV Cache**: Use cache-aware routing for optimal performance
4. **Route Intelligently**: Use multi-factor scoring for worker selection
5. **Use QoS to Maintain SLAs**: Implement early rejection and prioritization under load
6. **Scale Adaptively**: Use independent scaling for prefill and decode pools

This chapter provides comprehensive techniques for scaling disaggregated prefill and decode inference systems to serve large LLMs efficiently using modern GPU clusters and advanced networking.
