# Concurrent Streams Example

**Hardware:** Grace-Blackwell GB100/GB200  
**Software:** CUDA 13.0, C++17, Nsight Systems 2025.2.1, Nsight Compute 2024.3, Python 3.11, PyTorch nightly

## Build (C++)
```bash
make
```

## Run (C++)
```bash
./sequential_small_kernels
./concurrent_stream_kernels
```

## Run (Python)
```bash
python sequential_small_ops.py
python concurrent_streams.py
```

## Profile
```bash
nsys profile --trace=cuda --output concurrent_report ./concurrent_stream_kernels
ncu --set full --output concurrent_ncu_report ./concurrent_stream_kernels
```

### Sample Metrics

| Metric                  | Sequential | Concurrent |
|-------------------------|------------|------------|
| Kernel Launch Overhead  | High       | Low        |
| Throughput              | 50%        | 90%        |
| Stream Utilization      | N/A        | 2 streams  |
