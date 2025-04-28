#!/usr/bin/env python
"""
ucx_fragmentation_example.py

Shows how PyTorch's caching allocator can fragment GPU memory under
UCX/RDMA, exhausting registration pools and causing errors.
"""
import os
import time
import torch
import torch.distributed as dist

def log_mem(iteration):
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    print(f"[Iter {iteration:02d}] Reserved: {reserved/1e9:.3f} GB, "
          f"Allocated: {allocated/1e9:.3f} GB")

def main():
    # Initialize NCCL (over UCX if built accordingly)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Pre-allocate a large buffer once (registered by UCX)
    big_buf = torch.empty(int(2e8), dtype=torch.float32, device=local_rank)
    if rank == 0: log_mem(0)

    for i in range(1, 11):
        # Allocate and free varying sizes each iteration
        a = torch.randn(int(1e7), device=local_rank)   # ~40 MB
        b = torch.randn(int(5e7), device=local_rank)   # ~200 MB
        # Free tensors (cached by allocator)
        del a, b
        torch.cuda.synchronize()
        if rank == 0: log_mem(i)
        # Sync all ranks so logs stay ordered
        dist.barrier()
        time.sleep(0.1)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
