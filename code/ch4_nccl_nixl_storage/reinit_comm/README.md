# NCCL Communicator Reinitialization Example

This package demonstrates the performance pitfall of repeatedly initializing and destroying a NCCL process group every iteration (before), and the optimized version that initializes once (after). It includes profiling scripts to measure the overhead.

---

## Files Provided

- `before_reinit_comm.py`: Naive example reinitializing the process group each iteration.
- `after_reinit_comm.py`: Optimized example initializing the process group once.
- `requirements.txt`: Python dependencies.
- `run_nsys_before.sh` / `run_ncu_before.sh`: Profile the **before** version.
- `run_nsys_after.sh` / `run_ncu_after.sh`: Profile the **after** version.

---

## Requirements

- **Python:** 3.11  
- **PyTorch:** nightly 2.8.0+cu13  
- **CUDA Toolkit:** 13.0  
- **Profilers:**  
  - Nsight Systems 2025.2.1 (`nsys`)  
  - Nsight Compute 2024.3 (`ncu`)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Both scripts use PyTorch’s `mp.spawn` to launch 2 processes on 2 GPUs. Ensure you have 2 GPUs on your node.

```bash
# Before (inefficient)
python before_reinit_comm.py

# After (optimized)
python after_reinit_comm.py
```

---

## Profiling

### Before Version
```bash
./run_nsys_before.sh   # outputs nsys_before.qdrep
./run_ncu_before.sh    # outputs ncu_before_report.ncu-rep
```

### After Version
```bash
./run_nsys_after.sh    # outputs nsys_after.qdrep
./run_ncu_after.sh     # outputs ncu_after_report.ncu-rep
```

Open the `.qdrep` files in Nsight Systems to compare the per-iteration overhead, and the `.ncu-rep` in Nsight Compute for kernel-level details.
