# Chapter 18: Advanced Prefill-Decode and KV Cache Tuning

## Summary
These examples demonstrate faster decode and KV‑cache movement using FlashMLA/ThunderMLA kernels, FlexDecoding, and high‑bandwidth GPU‑to‑GPU transfers.

## Performance Takeaways
- Accelerate decode with FlashMLA/ThunderMLA kernel optimizations
- Employ FlexDecoding to compile optimized prefill/decode kernels per pattern
- Use NIXL RDMA for zero‑copy KV transfers and overlap with compute
- Apply PagedAttention and compression to increase KV capacity and efficiency
- Tune buffer sizes and overlap strategies to minimize hand‑off latency

Code examples demonstrating advanced decode kernels, KV cache optimizations, and fast GPU-to-GPU transfer techniques for disaggregated inference.

## Examples

- `flashmla_kernel.cu` - FlashMLA and ThunderMLA decode kernel implementations
- `flexdecoding_example.py` - PyTorch FlexDecoding for autoregressive inference
- `lmcache_config.yaml` - LMCache NIXL configuration for fast KV transfers

## Key Concepts

- **FlashMLA**: DeepSeek's optimized decode kernel for single-token generation
- **ThunderMLA**: Stanford's mega-kernel fusing attention + feedforward
- **FlexDecoding**: PyTorch's JIT-compiled kernels for flexible attention patterns
- **NIXL RDMA**: Fast GPU-to-GPU KV cache transfers with zero-copy
- **Disaggregated KV Cache**: Cluster-wide KV cache sharing and management
- **PagedAttention**: Efficient logical-to-physical KV block mapping

## Requirements

- CUDA 12.8+ with Ampere (SM 8.0) or newer for optimal FlashMLA performance
- PyTorch 2.8+ with flex_attention support
- NVIDIA NIXL library for RDMA transfers
- InfiniBand or NVLink for high-speed interconnects

## Usage

### FlashMLA Kernel
```bash
# Compile and run FlashMLA benchmark
nvcc -O3 -arch=sm_80 flashmla_kernel.cu -lcuda -lcublas -o flashmla_kernel
./flashmla_kernel
```

### FlexDecoding Demo
```bash
# Run FlexDecoding with different attention patterns
python flexdecoding_example.py

# Demonstrates:
# - Causal, local, and block-sparse attention patterns
# - Compiled prefill vs decode kernels
# - Jagged tensor support for variable lengths
# - PagedAttention integration
```

### LMCache Configuration
```bash
# Start prefill server with NIXL
CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=lmcache_config.yaml \
python -m vllm.entrypoints.api_server --model llama-70b --role prefill

# Start decode server
CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=lmcache_config.yaml \
python -m vllm.entrypoints.api_server --model llama-70b --role decode
```

## Performance Optimizations

### FlashMLA Benefits
- **20-35% faster decode** compared to standard kernels
- **Reduced memory bandwidth** through operation fusion
- **Warp-level optimizations** for efficient reductions
- **Vectorized memory access** with half4 operations

### FlexDecoding Features
- **Pattern flexibility** without custom CUDA code
- **Separate prefill/decode compilation** for optimal performance
- **Jagged tensor support** for batched variable-length sequences
- **PagedAttention integration** for memory efficiency

### KV Cache Transfer
- **5-10ms hand-off latency** for large prompts with NIXL RDMA
- **Zero-copy GPU-to-GPU** transfers over NVLink/InfiniBand
- **Overlap compute and transfer** for pipeline efficiency
- **1GB+ transfer buffers** for high-throughput scenarios

## Configuration

The LMCache YAML configuration supports:

```yaml
# NIXL RDMA setup
enable_nixl: true
nixl_buffer_size: 1073741824  # 1GB GPU buffer
nixl_buffer_device: "cuda"

# KV cache optimization
kv_cache_dtype: "fp16"        # or fp8 for compression
enable_prefix_caching: true
max_cached_tokens: 1048576

# Performance tuning
overlap_compute_transfer: true
use_flashmla: true
use_flexdecoding: true
```

## Hardware Requirements

### Optimal Performance
- **NVIDIA H100/B100** for FlashMLA Tensor Core optimizations
- **NVLink/NVSwitch** for intra-node GPU-to-GPU transfers
- **InfiniBand HDR** for inter-node RDMA with 200Gb/s+ bandwidth
- **High-bandwidth memory** (HBM3/4) for KV cache capacity

### Minimum Requirements
- **NVIDIA A100 or newer** (Ampere+ architecture)
- **PCIe 4.0** for reasonable GPU-to-GPU bandwidth
- **RDMA-capable NICs** for disaggregated deployments

## Profiling and Analysis

Use NVIDIA Nsight Systems to analyze:
- **Kernel fusion effectiveness** in FlashMLA/ThunderMLA
- **Memory bandwidth utilization** during KV transfers
- **Compute-transfer overlap** in disaggregated pipelines
- **Attention pattern efficiency** with FlexDecoding

Key regions to monitor:
- `flashmla_decode_kernel` execution time
- `NIXL_RDMA_Transfer` bandwidth and latency
- `flex_attention` compilation and runtime
- `KV_Cache_Update` memory patterns
