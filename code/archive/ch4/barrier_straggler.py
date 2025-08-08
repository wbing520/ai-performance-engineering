import os
import time
import torch
import torch.distributed as dist

def run(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    # Simulate some work (for demonstration, make rank 1 slower)
    if rank == 1:
        time.sleep(6)  # Simulate a straggler taking 6 seconds
    # ... (forward/backward work would go here) ...

    # Use a monitored barrier at end of iteration to detect laggards
    try:
        dist.monitored_barrier(timeout=5.0)
    except RuntimeError as e:
        print(f"Rank {rank} timed out at barrier: {e}")
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", 2))
    run(rank=int(os.getenv("LOCAL_RANK", 0)), world_size=world_size)
