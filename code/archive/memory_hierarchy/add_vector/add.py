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
#!/usr/bin/env python3
# add.py — PyTorch 2.8.0-nightly-cu130 + Triton 3.3.0
import torch, time, triton, triton.language as tl

@triton.jit
def triton_vecadd(A, B, C, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    tl.store(C + offs, a + b, mask=mask)

def run_triton(N=1<<20, BLOCK=1024):
    A = torch.arange(N, dtype=torch.float32, device='cuda')
    B = torch.arange(N, 0, -1, dtype=torch.float32, device='cuda')
    C = torch.empty_like(A)
    t0 = time.perf_counter()
    triton_vecadd[(N + BLOCK - 1) // BLOCK](A, B, C, N, BLOCK=BLOCK)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3

if __name__ == "__main__":
    N = 1<<20
    A = torch.arange(N)
    B = torch.arange(N, 0, -1)
    t0 = time.perf_counter(); C_cpu = A + B; t1 = time.perf_counter()
    A_gpu, B_gpu = A.cuda(), B.cuda()
    t2 = time.perf_counter(); C_gpu = A_gpu + B_gpu; torch.cuda.synchronize(); t3 = time.perf_counter()
    t_t = run_triton(N)
    print(f"CPU add     : {(t1-t0)*1e3:.3f} ms")
    print(f"PyTorch add : {(t3-t2)*1e3:.3f} ms")
    print(f"Triton add  : {t_t:.3f} ms")

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
