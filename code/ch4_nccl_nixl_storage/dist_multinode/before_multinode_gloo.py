#!/usr/bin/env python3
"""before_multinode_gloo.py
Multi-node all-reduce using Gloo (CPU-based).
"""

import argparse
import time
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True)
    parser.add_argument('--rank',       type=int, required=True)
    parser.add_argument('--master_addr',type=str, required=True)
    parser.add_argument('--master_port',type=int, required=True)
    args = parser.parse_args()

    dist.init_process_group(
        backend='gloo',
        world_size=args.world_size,
        rank=args.rank,
        init_method=f"tcp://{args.master_addr}:{args.master_port}")

    # 100 MB tensor
    num_elements = 100 * 1024 * 1024 // 4  # floats
    tensor = torch.ones(num_elements, dtype=torch.float32)

    # Warm-up
    for _ in range(3):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Timed all-reduce
    torch.cuda.synchronize()  # ensure clean state
    start = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start) * 1000

    print(f"Rank {args.rank} (Gloo): all-reduce 100 MB took {elapsed_ms:.2f} ms")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
