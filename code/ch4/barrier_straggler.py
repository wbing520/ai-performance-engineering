# barrier_straggler.py
import os
import torch
import torch.distributed as dist
import time

def run(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    
    # Simulate some work that might vary across ranks
    if rank == 1:
        # Simulate a straggler by adding extra delay
        time.sleep(2.0)
    
    # ... your forward/backward work here ...
    # For demo purposes, just do a simple operation
    dummy_tensor = torch.randn(1000, 1000, device=f"cuda:{local_rank}")
    result = torch.mm(dummy_tensor, dummy_tensor.t())
    
    # Before syncing at end of iteration, use a monitored barrier:
    try:
        # Wait up to 30 seconds for all ranks—if one lags, you'll get a timeout on that rank
        start_time = time.time()
        dist.monitored_barrier(timeout=30.0)
        barrier_time = time.time() - start_time
        print(f"Rank {rank} completed barrier in {barrier_time:.2f}s")
    except RuntimeError as e:
        print(f"Rank {rank} timed out at barrier: {e}")
    
    # Now proceed knowing all ranks are roughly in sync
    dist.destroy_process_group()

def main():
    import torch.multiprocessing as mp
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
