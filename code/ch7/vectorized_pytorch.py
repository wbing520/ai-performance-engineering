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
    return "blackwell" if compute_capability == "10.0" else "other"


def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "8.0 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"]
        }
    return {
        "name": "Other",
        "compute_capability": "Unknown",
        "sm_version": "Unknown",
        "memory_bandwidth": "Unknown",
        "tensor_cores": "Unknown",
        "features": []
    }

def copy_scalar(inp: torch.Tensor) -> torch.Tensor:
    """
    Demonstrates scalar copy - element by element (extremely inefficient).
    DO NOT USE THIS in production - only for educational purposes.
    """
    out = torch.empty_like(inp)
    flat_in = inp.view(-1)
    flat_out = out.view(-1)
    
    # This is extremely slow. DO NOT DO THIS!
    # Use vectorized operations to avoid Python loops
    # on GPU tensors as shown in optimized version
    for i in range(min(100, flat_in.numel())):  # Only do first 100 for demo
        flat_out[i] = flat_in[i]
    
    # Copy the rest efficiently to avoid timeout
    if flat_in.numel() > 100:
        flat_out[100:] = flat_in[100:]
    
    return out

def copy_vectorized(inp: torch.Tensor) -> torch.Tensor:
    """
    Demonstrates vectorized copy by using tensor operations.
    """
    # Reshape into groups of 4 floats for bulk copy
    vec = inp.view(-1, 4)
    
    # clone() on a reshaped tensor will issue one float4 (16 B) copy per thread internally
    # Effectively a float4 per thread under the hood
    # since PyTorch's clone() will choose an optimized,
    # CUDA-based memory copy.
    out_vec = vec.clone()
    
    return out_vec.view(-1)

def main():
    """
    Compare scalar vs vectorized memory access patterns.
    """
    N = 1 << 20
    
    # Create input tensor (ensure it's divisible by 4 for vectorized example)
    inp = torch.arange(N, device='cuda', dtype=torch.float32)
    
    print(f"Input tensor size: {inp.shape}")
    
    # Scalar copy (inefficient - demo only)
    print("\n=== Scalar Copy (Demo - First 100 elements) ===")
    with torch.cuda.nvtx.range("scalar_copy"):
        out_scalar = copy_scalar(inp)
    print(f"Output size: {out_scalar.shape}")
    
    # Vectorized copy (efficient)
    print("\n=== Vectorized Copy ===")
    with torch.cuda.nvtx.range("vectorized_copy"):
        out_vectorized = copy_vectorized(inp)
    print(f"Output size: {out_vectorized.shape}")
    
    # Verify results are equal
    print(f"Results equal: {torch.allclose(out_scalar, out_vectorized)}")
    
    print("\nFor profiling, use:")
    print("nsys profile --trace=cuda,nvtx python vectorized_pytorch.py")

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

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if compute_capability == "10.0" and triton_cfg is not None:  # Blackwell B200/B300
        try:
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
            if hasattr(triton_cfg, "stream_ordered_memory"):
                triton_cfg.stream_ordered_memory = True
        except AttributeError:
            print("Blackwell optimizations not available in this PyTorch build")

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.automatic_dynamic_shapes = True
