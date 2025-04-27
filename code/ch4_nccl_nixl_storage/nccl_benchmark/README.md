# GPU Peer-to-Peer Bandwidth Benchmark

This example measures GPU peer-to-peer bandwidth (e.g., NVLink or PCIe) between two devices.

## Hardware & Software Requirements

- **GPU:** NVIDIA Grace-Blackwell (GB100/GB200) or H100 (Hopper)
- **CUDA Toolkit:** 13.0
- **C++ Standard:** C++17
- **Compiler:** `nvcc` (CUDA 13.0)
- **Profilers:**
  - Nsight Systems 2025.2.1 (`nsys`)
  - Nsight Compute 2024.3 (`ncu`)

## Build Instructions

```bash
cd interconnect_benchmark
make
```

Generates `p2p_bandwidth_bench`.

## Run Instructions

Run the benchmark executable:

```bash
./p2p_bandwidth_bench
```

Expected output (sample):

```
Size (MB)    Bandwidth (GB/s)
4            150.3
16           180.8
64           200.1
256          195.7
1024         185.2
```

## Profiling

### Nsight Systems

```bash
./run_nsys.sh
```

Generates `nsys_p2p.qdrep`.

### Nsight Compute

```bash
./run_ncu.sh
```

Generates `ncu_p2p_report.ncu-rep`.
