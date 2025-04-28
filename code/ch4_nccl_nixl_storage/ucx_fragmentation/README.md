# UCX Memory Fragmentation Example

This example shows how PyTorch's caching allocator can fragment GPU memory under UCX/RDMA, potentially exhausting registration pools.

## Files

- `ucx_fragmentation_example.py`: The main script demonstrating memory fragmentation.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.11
- PyTorch nightly 2.8.0+cu13

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

On a single node with 2 GPUs and UCX support:

```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29502
torchrun --nnodes=1 --nproc_per_node=2 ucx_fragmentation.py
```

The script logs `torch.cuda.memory_reserved()` vs. `torch.cuda.memory_allocated()` to show cache growth.
