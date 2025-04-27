# Manual No-Overlap All-Reduce Example

This example shows how distributed training **without** overlapping communication and computation
performs the all-reduce **after** the backward pass, leading to a serialized communication phase.

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

1. Create a Python virtualenv or Conda env with Python 3.11.
2. Install dependencies:

   ```bash
   cd dist_no_overlap
   pip install -r requirements.txt
   ```

---

## Files Provided

- `before_no_overlap.py` — runnable PyTorch script
- `requirements.txt`    — Python dependencies
- `run_nsys.sh`         — Nsight Systems profiling script
- `run_ncu.sh`          — Nsight Compute profiling script

---

## How to Run

Launch **2** processes on GPUs 0 and 1:

```bash
python before_no_overlap.py --world_size 2
```

You should see something like:

```
Rank 0: iteration took 22.34 ms
```

---

## How to Profile

### Nsight Systems

```bash
./run_nsys.sh
```

Generates `nsys_nooverlap.qdrep`.

### Nsight Compute

```bash
./run_ncu.sh
```

Generates `nooverlap_ncu_report.ncu-rep`.

Inspect for a clear separation between backward and NCCL kernels (no overlap).
