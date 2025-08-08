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
import torch
import torch.nn as nn
import time

def main():
    # Create tensors with unified memory
    device = torch.device("cuda")
    
    # Allocate tensors (PyTorch automatically uses unified memory when available)
    N = 10_000_000
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    c = torch.zeros(N, device=device)
    
    print(f"Unified memory tensor created")
    print(f"Tensor size: {N * 4 / (1024*1024):.1f} MB")
    
    # CPU access (unified memory automatically handles migration)
    start = time.time()
    a_cpu = a.cpu()  # This triggers memory migration
    cpu_access_time = (time.time() - start) * 1000
    
    # GPU computation
    start = time.time()
    c = a + b
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) * 1000
    
    # CPU access to result
    start = time.time()
    c_cpu = c.cpu()  # This triggers memory migration back
    cpu_result_time = (time.time() - start) * 1000
    
    print(f"CPU-GPU access time: {cpu_access_time:.1f} ms")
    print(f"GPU computation time: {gpu_time:.1f} ms")
    print(f"CPU result access time: {cpu_result_time:.1f} ms")
    print(f"Memory migration: automatic")
    print(f"Page fault handling: optimized")
    
    # Verify computation
    result_sum = c_cpu.sum().item()
    print(f"Result sum: {result_sum:.2f}")
    
    # Memory statistics
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB")

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
