# DDP Overlap All-Reduce Example

This example uses PyTorch’s `DistributedDataParallel` (DDP) to overlap gradient
communication with the backward computation. You should observe a sawtooth pattern
of interleaved compute and NCCL work and a shorter iteration time.

---

## Hardware & Software Requirements

- **GPU**: NVIDIA Grace-Blackwell (GB100/GB200) or fallback H100 (Hopper)
- **CUDA Toolkit**: 13.0
- **Python**: 3.11
- **PyTorch**: nightly **2.8.0+cu13**
- **Profilers**:
  - Nsight Systems 2025.2.1 (`nsys`)
  - Nsight Compute 2024.3 (`ncu`)

---

## Setup

```bash
cd dist_ddp_overlap
pip install -r requirements.txt
```

---

## Files Provided

- `after_overlap_ddp.py` — PyTorch DDP script
- `requirements.txt`     — Python deps
- `run_nsys.sh`          — Nsight Systems script
- `run_ncu.sh`           — Nsight Compute script

---

## How to Run

```bash
python after_overlap_ddp.py --world_size 2
```

Expected output:

```
Rank 0: iteration took 15.12 ms
```

---

## How to Profile

### Nsight Systems

```bash
./run_nsys.sh
```
Generates `nsys_ddp.qdrep`.

### Nsight Compute

```bash
./run_ncu.sh
```
Generates `ddp_ncu_report.ncu-rep`.

Inspect interleaving of compute and NCCL kernels in the system trace.
