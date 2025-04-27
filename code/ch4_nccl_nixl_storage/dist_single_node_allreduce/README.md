# Single-Node All-Reduce: Gloo vs. NCCL

This example measures and compares a large (≈400 MB) all-reduce on a single server with 2 GPUs using:

- **Gloo** backend: CPU-mediated over localhost TCP  
- **NCCL** backend: GPU-direct P2P/RDMA  

You will observe a dramatic latency and throughput difference between the two.

---

## Hardware & Software Requirements

- **Node:** 1 server with 2 NVIDIA GPUs  
- **GPU:** GB100/GB200 (Grace-Blackwell) or fallback H100 (Hopper)  
- **Network:** localhost or IPoIB (for Gloo); NVLink or P2P for NCCL  
- **CUDA Toolkit:** 13.0  
- **Python:** 3.11  
- **PyTorch:** nightly 2.8.0+cu13  
- **Profilers:**  
  - Nsight Systems 2025.2.1 (`nsys`)  
  - Nsight Compute 2024.3 (`ncu`)

---

## Setup

On the single node:

```bash
git clone <this-repo>/dist_single_node_allreduce
cd dist_single_node_allreduce
pip install -r requirements.txt
```

---

## Files Provided

- `allreduce_gloo_single_node.py`  
- `allreduce_nccl_single_node.py`  
- `requirements.txt`  
- `run_nsys_nccl_single_node.sh` / `run_ncu_nccl_single_node.sh`  
- `run_nsys_gloo_single_node.sh` / `run_ncu_gloo_single_node.sh`  

---

## How to Run (Single-Node, 2 GPUs)

We use `torchrun` to spawn 2 processes on 1 node. Only **one** rendezvous port is needed:

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
```

### Gloo (CPU+TCP)

```bash
torchrun   --nnodes=1   --nproc_per_node=2   allreduce_gloo_single_node.py
```

### NCCL (GPU-direct P2P/RDMA)

```bash
export NCCL_SOCKET_IFNAME=ib0   # only needed if bootstrapping over RDMA
torchrun   --nnodes=1   --nproc_per_node=2   allreduce_nccl_single_node.py
```

---

## Expected Output

- **Gloo** (CPU+TCP):  
  ```
  Rank0: Gloo all-reduce of 400.0 MB took ~180.0 ms   (≈2.2 GB/s)
  ```

- **NCCL** (GPU-direct P2P/RDMA):  
  ```
  Rank0: All-reduce of 400.0 MB took ~4.0 ms   (≈100 GB/s)
  ```

---

## Profiling the NCCL Run

Once the NCCL script is running, in a separate shell:

```bash
./run_nsys_nccl_single_node.sh   # produces nsys_nccl_allreduce_single_node.qdrep
./run_ncu_nccl_single_node.sh    # produces ncu_nccl_allreduce_single_node_report.ncu-rep
```

Open the `.qdrep` in Nsight Systems to inspect compute vs. communication overlap, or load `.ncu-rep` in Nsight Compute for kernel metrics.

---

## Profiling the Gloo Run

To capture CPU-side traces:

```bash
./run_nsys_gloo_single_node.sh   # produces nsys_gloo_allreduce_single_node.qdrep
./run_ncu_gloo_single_node.sh    # minimal GPU kernels, but report generated
```

This will highlight the CPU and TCP overhead that Gloo incurs compared to NCCL.

---

With these instructions, you can easily benchmark and profile on **one machine with two GPUs**, and see first-hand the speedup of GPU-direct NCCL over a CPU-bound Gloo fallback.
