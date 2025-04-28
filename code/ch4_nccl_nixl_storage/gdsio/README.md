# GPUDirect Storage (GDS) Benchmark Example

This example uses NVIDIA's `gdsio` tool to compare storage throughput for CPU-based reads versus GPUDirect Storage into GPU memory.

## Hardware & Software Requirements

- **GPUs:** GB100/GB200 (Grace-Blackwell) or H100 (Hopper)  
- **CUDA Toolkit:** 13.0 or later (with GDS enabled)  
- **gdsio Tool:** Located under `/usr/local/cuda-13.0/gds/tools/gdsio`  
- **Storage:** Large file accessible at `/mnt/data/large_file` (~>10 GiB)  
- **Profilers:** Nsight Systems / Compute (optional)

## Before & After Commands

```bash
# Before: CPU-mediated read path (-x 1)
./run_gdsio_before.sh
```

```bash
# After: Direct GPU path via GDS (-x 0)
./run_gdsio_after.sh
```

## Sample Output

```
# CPU Path
Total Throughput: 8.0 GB/s
Average Latency: 1.25 ms

# GPU Path (GDS)
Total Throughput: 9.6 GB/s
Average Latency: 1.00 ms
```

## Performance Comparison

| Path                          | Throughput       | Latency      |
|-------------------------------|-----------------:|-------------:|
| Storage → CPU (`-x 1`)        | 8.0 GB/s         | 1.25 ms      |
| Storage → GPU (`-x 0`, GDS)   | 9.6 GB/s (+20%)  | 1.00 ms (–20%) |

This benchmark shows a 20% increase in throughput and 20% decrease in latency when using GPUDirect Storage, while freeing CPU resources for other work.
