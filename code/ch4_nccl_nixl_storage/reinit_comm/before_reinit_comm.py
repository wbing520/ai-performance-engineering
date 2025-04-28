#!/usr/bin/env python
# before_reinit_comm.py
# Demonstrates naive init/destroy per iteration

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, world_size):
    torch.cuda.set_device(rank)
    for i in range(5):
        # Naively create and destroy communicator each iteration
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:45678",
            world_size=world_size,
            rank=rank
        )
        tensor = torch.ones(1).cuda(rank)
        # Warm-up sync
        torch.cuda.synchronize(rank)
        start = time.time()
        dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        if rank == 0:
            print(f"Iter {i}: all-reduce took {(time.time()-start)*1000:.2f} ms")
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
