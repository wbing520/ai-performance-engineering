# Multi-Node All-Reduce: Gloo vs. NCCL

This example measures and compares a large (≈400 MB) all-reduce across two separate nodes (2 GPUs, 1 per node) using:

- **Gloo** backend: CPU-mediated over TCP  
- **NCCL** backend: GPU-direct RDMA over InfiniBand/RoCE  

You will observe a dramatic latency and throughput difference between the two.

---

## Hardware & Software Requirements

- **Nodes:** 2 servers, each with 1 NVIDIA GPU  
- **GPU:** GB100/GB200 (Grace-Blackwell) or fallback H100 (Hopper)  
- **Network:** InfiniBand HDR 100 Gb/s (or 100 GbE)  
- **CUDA Toolkit:** 13.0  
- **Python:** 3.11  
- **PyTorch:** nightly 2.8.0+cu13  
- **Profilers:**  
  - Nsight Systems 2025.2.1 (`nsys`)  
  - Nsight Compute 2024.3 (`ncu`)

---

## Setup

On **both** nodes:

```bash
git clone <this-repo>/dist_multinode_allreduce
cd dist_multinode_allreduce
pip install -r requirements.txt
```

Ensure both nodes can reach each other on TCP port **29502**.

---

## Files Provided

- `allreduce_gloo_multi_node.py`  
- `allreduce_nccl_multi_node.py`  
- `requirements.txt`  
- `run_nsys_nccl_multi_node.sh` / `run_ncu_nccl_multi_node.sh`  
- `run_nsys_gloo_multi_node.sh` / `run_ncu_gloo_multi_node.sh`  

---

## How to Run

### Gloo (CPU+TCP)

On **Node 0** (rank 0, IP 10.0.0.1):

```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29502
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 allreduce_gloo_multi_node.py
```

On **Node 1** (rank 1, IP 10.0.0.2):

```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29502
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 allreduce_gloo_multi_node.py
```

### NCCL (GPU-direct RDMA)

On **Node 0**:

```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29502
export NCCL_SOCKET_IFNAME=ib0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 allreduce_nccl_multi_node.py
```

On **Node 1**:

```bash
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29502
export NCCL_SOCKET_IFNAME=ib0
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 allreduce_nccl_multi_node.py
```

---

## Expected Output

- **Gloo**:  
  ```
  Rank0: Gloo all-reduce of 400.0 MB took ~180.0 ms   (≈2.2 GB/s)
  ```

- **NCCL**:  
  ```
  Rank0: All-reduce of 400.0 MB took ~5.0 ms   (≈80 GB/s)
  ```

---

## Profiling

**NCCL**:

```bash
./run_nsys_nccl_multi_node.sh   # generates nsys_nccl_allreduce_multi_node.qdrep
./run_ncu_nccl_multi_node.sh    # generates ncu_nccl_allreduce_multi_node_report.ncu-rep
```

**Gloo**:

```bash
./run_nsys_gloo_multi_node.sh   # generates nsys_gloo_allreduce._multi_nodeqdrep
./run_ncu_gloo_multi_node.sh    # generates ncu_gloo_allreduce_multi_node_report.ncu-rep
```
