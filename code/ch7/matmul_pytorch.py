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

def naive_matmul(A, B):
    """
    Naive matrix multiplication - extremely inefficient for educational purposes.
    Demonstrates redundant global memory access patterns.
    """
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    
    # This is extremely slow - only do a small subset for demo
    max_iter = min(32, N)  # Limit to avoid timeout
    
    for i in range(max_iter):
        for j in range(max_iter):
            # Each dot product loads A[i,:] and B[:,j] from global memory repeatedly
            C[i, j] = (A[i, :] * B[:, j]).sum()
    
    # Use efficient operation for the rest to complete the computation
    if max_iter < N:
        C[max_iter:, :] = torch.mm(A[max_iter:, :], B)
        C[:max_iter, max_iter:] = torch.mm(A[:max_iter, :], B[:, max_iter:])
    
    return C

def tiled_matmul(A, B, tile_size=32):
    """
    Tiled matrix multiplication to demonstrate data reuse patterns.
    PyTorch's torch.mm already implements efficient tiling internally.
    """
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            C_block = torch.zeros((tile_size, tile_size), device='cuda')
            
            for k in range(0, N, tile_size):
                # Define tile boundaries
                i_end = min(i + tile_size, N)
                j_end = min(j + tile_size, N)
                k_end = min(k + tile_size, N)
                
                A_block = A[i:i_end, k:k_end]
                B_block = B[k:k_end, j:j_end]
                
                # torch.mm uses an optimized kernel (likely tiling internally)
                C_block_partial = torch.mm(A_block, B_block)
                
                # Accumulate into the correct portion of C_block
                actual_rows = i_end - i
                actual_cols = j_end - j
                C_block[:actual_rows, :actual_cols] += C_block_partial
            
            # Copy result to output matrix
            C[i:i+tile_size, j:j+tile_size] = C_block
    
    return C

def optimized_matmul(A, B):
    """
    Use PyTorch's optimized matrix multiplication.
    This leverages cuBLAS/CUTLASS which implement advanced optimizations.
    """
    return torch.mm(A, B)

def main():
    """
    Compare different matrix multiplication approaches.
    """
    N = 1024
    
    # Create test matrices
    A = torch.ones((N, N), device='cuda', dtype=torch.float32)
    B = torch.ones((N, N), device='cuda', dtype=torch.float32)
    
    print(f"Matrix size: {N}x{N}")
    
    # Naive implementation (partial for demo)
    print("\n=== Naive MatMul (Partial - 32x32 subset) ===")
    with torch.cuda.nvtx.range("naive_matmul"):
        C_naive = naive_matmul(A, B)
    print(f"Result shape: {C_naive.shape}")
    print(f"Sample result C[0,0]: {C_naive[0,0].item()}")
    
    # Tiled implementation
    print("\n=== Tiled MatMul ===")
    with torch.cuda.nvtx.range("tiled_matmul"):
        C_tiled = tiled_matmul(A, B)
    print(f"Result shape: {C_tiled.shape}")
    print(f"Sample result C[0,0]: {C_tiled[0,0].item()}")
    
    # Optimized PyTorch implementation
    print("\n=== Optimized MatMul (PyTorch) ===")
    with torch.cuda.nvtx.range("optimized_matmul"):
        C_optimized = optimized_matmul(A, B)
    print(f"Result shape: {C_optimized.shape}")
    print(f"Sample result C[0,0]: {C_optimized[0,0].item()}")
    
    # Verify correctness (should all be N since A and B are all ones)
    expected_value = float(N)
    print(f"\nExpected result: {expected_value}")
    print(f"Optimized result close to expected: {torch.allclose(C_optimized, torch.full_like(C_optimized, expected_value))}")
    
    print("\nFor profiling, use:")
    print("nsys profile --trace=cuda,nvtx python matmul_pytorch.py")

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
