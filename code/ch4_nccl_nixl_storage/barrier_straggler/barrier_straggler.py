#!/usr/bin/env python
"""
barrier_straggler_example.py

Demonstrates using torch.distributed.monitored_barrier() to catch slow ranks
(stragglers) in a two-process DDP setup.
"""
import os
import time
import argparse

import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Straggler detection demo")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Seconds to wait at barrier before timeout")
    args = parser.parse_args()

    # Initialize process group (env:// picks up MASTER_ADDR, MASTER_PORT, LOCAL_RANK)
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Simulate work: rank 1 sleeps longer
    work_time = 1.0 if rank == 0 else 5.0
    print(f"[Rank {rank}] Starting work for {work_time}s...")
    time.sleep(work_time)
    print(f"[Rank {rank}] Work done, entering barrier...")

    # Monitored barrier will raise on slowness
    try:
        dist.monitored_barrier(timeout=args.timeout)
        print(f"[Rank {rank}] Passed barrier successfully.")
    except RuntimeError as e:
        print(f"[Rank {rank}] Barrier timeout: {e}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
