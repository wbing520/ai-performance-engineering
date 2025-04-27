# NIXL Asynchronous GPU→GPU Transfer Example

This example shows how to use NVIDIA’s NIXL to move large buffers
between two GPUs asynchronously, overlapping transfer with computation.

---

## Hardware & Software Requirements

- **GPU:** NVIDIA Grace-Blackwell (GB100/GB200) or H100 (Hopper)
- **CUDA Toolkit:** 13.0
- **C++ Standard:** C++17
- **Profilers:** Nsight Systems 2025.2.1, Nsight Compute 2024.3
- **Dependencies:** NVIDIA NIXL library, CUDA runtime

---

## Build

```bash
cd nixl_transfer
make
```

Generates `nixl_transfer`.

---

## Run

```bash
./nixl_transfer
```

Expected output:

```
Transfer posted, running concurrently...
Computation done.
Transfer completed in XX.XX ms
Result verified!
```

---

## Profile

### Nsight Systems

```bash
./run_nsys.sh
```

Produces `nsys_nixl.qdrep`.

### Nsight Compute

```bash
./run_ncu.sh
```

Produces `ncu_nixl_report.ncu-rep`.
