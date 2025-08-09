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
            "memory_bandwidth": "8.0 TB/s",
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
# add_sequential.py
# Naive PyTorch code that performs sequential GPU operations

import torch
import time

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A
C = torch.empty_like(A)

# Ensure all previous work is done
torch.cuda.synchronize()

start_time = time.time()

# Naive, Sequential GPU operations - DO NOT DO THIS
for i in range(N):
    C[i] = A[i] + B[i]  # This launches N tiny GPU operations serially

torch.cuda.synchronize()
elapsed_time = (time.time() - start_time) * 1000

print(f"Sequential PyTorch time: {elapsed_time:.2f} ms")
print(f"Result: C[0] = {C[0]}, C[N-1] = {C[N-1]}")

# Architecture-specific information
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    print(f"GPU: {device_props.name}")
    print(f"Compute Capability: {compute_capability}")
    
    if compute_capability == "9.0":  # Hopper H100/H200
        print("Architecture: Hopper H100/H200")
    elif compute_capability == "10.0":  # Blackwell B200/B300
        print("Architecture: Blackwell B200/B300")
    else:
        print(f"Architecture: Other (Compute Capability {compute_capability})")
