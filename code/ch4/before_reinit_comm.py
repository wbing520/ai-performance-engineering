# before_reinit_comm.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, world_size):
    torch.cuda.set_device(rank)
    
    for i in range(5):  # simulate 5 iterations
        # This naive approach re-initializes NCCL each
        # iteration. THIS IS EXTREMELY SLOW AND NOT
        # RECOMMENDED!!!
        start_init = time.time()
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:45678",
                               world_size=world_size, rank=rank)
        init_time = time.time() - start_init
        
        # do a tiny all-reduce to simulate some work
        tensor = torch.ones(1).cuda(rank)
        dist.all_reduce(tensor)
        
        if rank == 0:
            print(f"Iter {i} done (init took {init_time*1000:.2f} ms)")
        
        dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
