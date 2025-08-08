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

def uncoalesced_copy(input_tensor, stride):
    """
    Demonstrates uncoalesced memory access pattern using strided indexing.
    This creates a gather operation that causes uncoalesced loads.
    """
    # Flatten to 1D so we know exactly which dimension we're indexing
    flat_tensor = input_tensor.contiguous().view(-1)
    
    # Generate indices with a fixed stride to gather
    idx = torch.arange(0, flat_tensor.numel(), stride,
                      device=flat_tensor.device, dtype=torch.long)
    
    # index_select uses a gather kernel that issues uncoalesced loads
    return torch.index_select(flat_tensor, 0, idx)

def coalesced_copy(input_tensor):
    """
    Demonstrates coalesced memory access - PyTorch handles this efficiently.
    """
    # PyTorch's clone() operation is already optimized for coalesced access
    return input_tensor.clone()

def main():
    """
    Compare uncoalesced vs coalesced memory access patterns.
    """
    n, stride = 1 << 20, 2
    
    # Create input tensor
    inp = torch.arange(n * stride, device='cuda', dtype=torch.float32)
    
    print(f"Input tensor size: {inp.shape}")
    print(f"Stride: {stride}")
    
    # Uncoalesced access (inefficient)
    print("\n=== Uncoalesced Copy ===")
    with torch.cuda.nvtx.range("uncoalesced_copy"):
        out_uncoalesced = uncoalesced_copy(inp, stride)
    print(f"Output size: {out_uncoalesced.shape}")
    
    # Coalesced access (efficient)
    print("\n=== Coalesced Copy ===")
    with torch.cuda.nvtx.range("coalesced_copy"):
        out_coalesced = coalesced_copy(inp)
    print(f"Output size: {out_coalesced.shape}")
    
    print("\nFor profiling, use:")
    print("nsys profile --trace=cuda,nvtx python memory_access_pytorch.py")

if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        exit(1)
    
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
