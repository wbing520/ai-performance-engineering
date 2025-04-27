# Multi-Node All-Reduce: Gloo vs. NCCL

This example compares distributed all-reduce across 2 nodes using:

- **Gloo backend** (no GPU-direct RDMA, CPU-mediated)  
- **NCCL backend** (GPU-direct RDMA for faster throughput)

You’ll see the latency difference when running the same 100 MB all-reduce on 2 GPUs across two machines.

---

## Hardware & Software Requirements

- **Nodes**: 2 separate servers, each with 1 NVIDIA GPU  
  - GPU: GB100/GB200 (Grace-Blackwell) or fallback H100 (Hopper)  
- **Network**: InfiniBand HDR 100 Gb/s (or 100 GbE)  
- **CUDA Toolkit**: 13.0  
- **Python**: 3.11  
- **PyTorch**: nightly **2.8.0+cu13**  
- **Profilers**:  
  - Nsight Systems 2025.2.1 (`nsys`)  
  - Nsight Compute 2024.3 (`ncu`)  

---

## Setup

1. On **both** nodes, clone this folder and install dependencies:

   ```bash
   cd dist_multinode
   pip install -r requirements.txt
   ```

2. Ensure both machines can reach each other on port **29502** TCP.

---

## Files Provided

- `before_multinode_gloo.py`  — PyTorch all-reduce over Gloo  
- `after_multinode_nccl.py`   — PyTorch all-reduce over NCCL  
- `requirements.txt`           — Python deps  
- `run_nsys.sh`                — Profiles the NCCL run via Nsight Systems  
- `run_ncu.sh`                 — Profiles the NCCL run via Nsight Compute  

---

## How to Run

Each node runs the same command, specifying its rank (0 or 1) and the other node’s IP:

```bash
# On Node 0 (IP=10.0.0.1)
python before_multinode_gloo.py --world_size 2 --rank 0 --master_addr 10.0.0.1 --master_port 29502

# On Node 1 (IP=10.0.0.2)
python before_multinode_gloo.py --world_size 2 --rank 1 --master_addr 10.0.0.1 --master_port 29502
```

Repeat the same for `after_multinode_nccl.py` to compare.

Expected output (Gloo):

```
Rank 0 (Gloo): all-reduce 100 MB took 180.23 ms
```

Expected output (NCCL):

```
Rank 0 (NCCL): all-reduce 100 MB took 45.67 ms
```

---

## How to Profile (NCCL run)

**On either node**, after starting the NCCL script once (background), run:

```bash
./run_nsys.sh
./run_ncu.sh
```

This will attach to the process and capture traces/reports.
