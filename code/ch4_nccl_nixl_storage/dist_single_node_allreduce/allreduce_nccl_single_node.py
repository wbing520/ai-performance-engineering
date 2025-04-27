#!/usr/bin/env python

import os
import time
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Single-node NCCL all-reduce benchmark")
    parser.add_argument(
        "--data-size",
        type=int,
        default=1024 * 1024 * 100,  # 100M floats ≈ 400 MB
    help="Number of elements per tensor",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    tensor = torch.ones(args.data_size, dtype=torch.float32, device=f"cuda:{local_rank}")

    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        start = time.time()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    if rank == 0:
        elapsed = time.time() - start
        mb = args.data_size * 4 / 1e6
        print(f"Rank0: All-reduce of {mb:.1f} MB took {elapsed*1000:.2f} ms "
              f"({mb/elapsed/1e3:.1f} GB/s)")

    if rank == 0:
        assert tensor[0].item() == world_size

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
