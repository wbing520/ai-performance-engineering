# after_reinit_comm.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def run(rank, world_size):
    # Pin GPU
    torch.cuda.set_device(rank)
    
    # Initialize NCCL communicator once
    start_init = time.time()
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:45678",
        world_size=world_size,
        rank=rank
    )
    init_time = time.time() - start_init
    
    if rank == 0:
        print(f"One-time initialization took {init_time*1000:.2f} ms")
    
    # Now run iterations with the same communicator
    for i in range(5):
        start_iter = time.time()
        
        # do a tiny all-reduce to simulate some work
        tensor = torch.ones(1).cuda(rank)
        dist.all_reduce(tensor)
        
        iter_time = time.time() - start_iter
        
        if rank == 0:
            print(f"Iter {i} done (all-reduce took {iter_time*1000:.2f} ms)")
    
    # Clean up once at the end
    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
