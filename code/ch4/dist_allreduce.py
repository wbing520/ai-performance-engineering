#!/usr/bin/env python
# dist_allreduce.py

import os
import time
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Multi-node Gloo all-reduce benchmark")
    parser.add_argument(
        "--data-size",
        type=int,
        default=1024 * 1024 * 100,  # 100M floats ≈ 400 MB
        help="Number of elements in the tensor",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        choices=["gloo", "nccl"],
        help="Backend to use for distributed communication"
    )
    args = parser.parse_args()

    # Initialize the default ProcessGroup over env:// (uses MASTER_ADDR, MASTER_PORT, etc.)
    dist.init_process_group(backend=args.backend, init_method="env://")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Allocate a large tensor
    device = "cuda" if args.backend == "nccl" else "cpu"
    tensor = torch.ones(args.data_size, dtype=torch.float32, device=device)
    
    # Warm up and barrier
    dist.barrier()
    
    if rank == 0:
        start = time.time()
    
    # All-reduce (sum) across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Barrier + timing
    dist.barrier()
    
    if rank == 0:
        elapsed = time.time() - start
        mb = args.data_size * 4 / 1e6
        print(f"Rank0: All-reduce of {mb:.1f} MB took {elapsed*1000:.2f} ms "
              f"({mb/elapsed/1e3:.1f} GB/s)")
    
    # Verify correctness on rank 0
    if rank == 0:
        expected_value = world_size  # sum of 1's across all ranks
        actual_value = tensor[0].item()
        assert abs(actual_value - expected_value) < 1e-6, f"Expected {expected_value}, got {actual_value}"
        print("✓ All-reduce correctness verified")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
