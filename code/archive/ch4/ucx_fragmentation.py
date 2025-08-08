import torch
import torch.distributed as dist
import time
import os

def log_mem(iteration):
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    print(f"[Iter {iteration:02d}] Reserved: {reserved/1e9:.3f} GB, Allocated: {allocated/1e9:.3f} GB")

def run(rank, world_size):
    # Initialize DDP (for a real multi-GPU scenario)
    dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=rank)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    # Allocate a large buffer once (e.g., ~0.8 GB)
    big_buffer = torch.empty(int(2e8), device="cuda")  # 200e6 floats ~ 0.8 GB
    log_mem(0)

    for i in range(1, 11):
        # Simulate per-iteration allocations
        small = torch.randn(int(1e7), device="cuda")   # ~40 MB
        medium = torch.randn(int(5e7), device="cuda")  # ~200 MB

        # Free them to return to allocator cache
        del small, medium
        torch.cuda.synchronize()

        # Log memory after freeing
        log_mem(i)
        # Barrier to sync prints across ranks (if multi-GPU)
        dist.barrier()
        time.sleep(0.1)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = int(os.getenv("WORLD_SIZE", 1))
    run(rank=int(os.getenv("LOCAL_RANK", 0)), world_size=world_size)
