import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Architecture detection
    if compute_capability == "9.0":
        return "hopper"  # H100/H200
    elif compute_capability == "10.0":
        return "blackwell"  # B200/B300
    else:
        return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "hopper":
        return {
            "name": "Hopper H100/H200",
            "compute_capability": "9.0",
            "sm_version": "sm_90",
            "memory_bandwidth": "3.35 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3", "Transformer Engine", "Dynamic Programming"]
        }
    elif arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "3.2 TB/s",
            "tensor_cores": "4th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    else:
        return {
            "name": "Other",
            "compute_capability": "Unknown",
            "sm_version": "Unknown",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Unknown",
            "features": []
        }
#!/usr/bin/env python
import os
import time
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Multi-node all-reduce benchmark")
    parser.add_argument("--data-size", type=int, default=1024*1024*100,  # 100M floats ≈ 400 MB
                        help="Number of float elements in the tensor")
    args = parser.parse_args()

    # Initialize default ProcessGroup (env:// uses env vars like MASTER_ADDR/PORT, etc.)
    dist.init_process_group(backend="gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Allocate a large tensor on GPU for all-reduce
    tensor = torch.ones(args.data_size, dtype=torch.float32, device="cuda")

    # Synchronize and measure all-reduce
    dist.barrier()
    if rank == 0:
        start = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    if rank == 0:
        elapsed = time.time() - start
        mb = args.data_size * 4 / 1e6
        print(f"Rank0: All-reduce of {mb:.1f} MB took {elapsed*1000:.2f} ms  "
              f"({mb/elapsed/1e3:.1f} GB/s)")
        # Example expected output with Gloo: ~200 ms (2 GB/s) for 400 MB
        # With NCCL and RDMA: ~4 ms (100 GB/s)

    # Verify correctness (each element should equal world_size after SUM)
    if rank == 0:
        assert torch.allclose(tensor[0], torch.tensor(float(world_size))), "Result incorrect"

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    if compute_capability == "9.0":  # Hopper H100/H200
        torch._inductor.config.triton.use_hopper_optimizations = True
        torch._inductor.config.triton.hbm3_optimizations = True
    elif compute_capability == "10.0":  # Blackwell B200/B300
        torch._inductor.config.triton.use_blackwell_optimizations = True
        torch._inductor.config.triton.hbm3e_optimizations = True
        torch._inductor.config.triton.tma_support = True
    
    # Enable latest PyTorch 2.8 features
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.autotune_mode = "max-autotune"
    torch._dynamo.config.automatic_dynamic_shapes = True
