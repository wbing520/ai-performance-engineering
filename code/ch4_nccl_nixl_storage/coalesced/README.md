# Coalesced Global Memory Access Example

This example demonstrates a simple coalesced copy kernel in CUDA and shows
how to profile it with Nsight Systems (`nsys`) and Nsight Compute (`ncu`).

---

## Hardware & Software Requirements

- **GPU**: NVIDIA Grace-Blackwell (GB100/GB200) or fallback to H100 (Hopper)
- **CUDA Toolkit**: 13.0
- **C++ Standard**: C++17
- **Compiler**: `nvcc` from CUDA 13.0
- **Profilers**:
  - Nsight Systems 2025.2.1 (`nsys`)
  - Nsight Compute 2024.3 (`ncu`)
- **Python**: (for PyTorch profiling examples in later sections)
  - Python 3.11
  - PyTorch nightly (2.8.0+cu13)

---

## Build Instructions

```bash
cd coalesced
make
```

The `Makefile` will compile `coalesced_copy.cu` into the executable `coalesced_copy`.

---

## Run Example

This just runs the kernel and verifies it:

```bash
./coalesced_copy
```

You should see no output on success.

---

## Profiling

### Nsight Systems

```bash
./run_nsys.sh
```

This runs:

```bash
nsys profile   --sample none   --trace=cuda,cudnn   --output nsys_coalesced   ./coalesced_copy
```

### Nsight Compute

```bash
./run_ncu.sh
```

This runs:

```bash
ncu --target-processes all     --set full     --launch-skip 0     --launch-count 1     --output coalesced_ncu_report     ./coalesced_copy
```

Check the generated `nsys_coalesced.qdrep` and `coalesced_ncu_report.ncu-rep`
with the NVIDIA tools to inspect metrics like `gld_efficiency`, `achieved_occupancy`, etc.
