import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def run(rank, world_size):
    torch.cuda.set_device(rank)
    # Initialize NCCL communicator once at the start
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:45678",
                             world_size=world_size, rank=rank)
    # Simulate 5 training iterations
    for i in range(5):
        torch.cuda.synchronize(rank)
        start = time.time() if rank == 0 else None
        tensor = torch.ones(1, device=f"cuda:{rank}")
        dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        if rank == 0:
            elapsed_ms = (time.time() - start) * 1000
            print(f"Iter {i} all-reduce time: {elapsed_ms:.2f} ms")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)
