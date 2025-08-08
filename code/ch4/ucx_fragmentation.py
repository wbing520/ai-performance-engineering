# ucx_fragmentation.py
import torch
import torch.distributed as dist
import time
import os

def log_mem(iteration):
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    print(f"[Iter {iteration:02d}] Reserved: {reserved/1e9:.3f} GB, "
          f"Allocated: {allocated/1e9:.3f} GB")

def run(rank, world_size):
    # Standard DDP / UCX init
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    
    # Pre-allocate a big buffer that UCX will register once and hold
    big_buffer = torch.empty(int(2e8), device=local_rank)  # ~0.8 GB
    log_mem(0)
    
    for i in range(10):
        # Simulate variable-size allocations
        small_tensor = torch.randn(1000 + i * 100, 1000, device=local_rank)
        
        # All-reduce to trigger UCX/RDMA registration
        dist.all_reduce(small_tensor)
        
        # Clear cache periodically to avoid fragmentation
        if i % 3 == 0:
            torch.cuda.empty_cache()
        
        log_mem(i + 1)
        
        # Important: del to release reference
        del small_tensor
    
    # Final cleanup
    dist.destroy_process_group()

def main():
    import torch.multiprocessing as mp
    world_size = 2
    
    # Set environment variables for UCX (if using)
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "3")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    
    mp.spawn(run, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
