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
"""
Chapter 14: OpenAI Triton Kernel Development
Comprehensive examples demonstrating Triton kernel writing, autotuning, and PyTorch integration
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

def demonstrate_basic_triton_kernel():
    """Demonstrate basic Triton kernel for vector addition"""
    print("=== Basic Triton Kernel ===")
    
    BLOCK_SIZE = 1024
    
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)  # unique program ID for each block
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)  # each program handles BLOCK_SIZE elements
        
        # Create a mask to guard against out-of-bounds
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)  # masked load
        y = tl.load(y_ptr + offsets, mask=mask)
        result = x + y
        tl.store(out_ptr + offsets, result, mask=mask)  # masked store
    
    # Test the kernel
    n_elements = 10000
    x = torch.randn(n_elements, device='cuda')
    y = torch.randn(n_elements, device='cuda')
    out = torch.empty_like(x)
    
    # Launch kernel
    threads_per_block = 1024
    blocks_per_grid = triton.cdiv(n_elements, threads_per_block)
    
    vector_add_kernel[blocks_per_grid](
        x, y, out, n_elements, BLOCK_SIZE=threads_per_block
    )
    
    # Verify result
    expected = x + y
    if torch.allclose(out, expected):
        print("✓ Basic Triton kernel works correctly")
    else:
        print("✗ Basic Triton kernel failed")

def demonstrate_shared_memory_kernel():
    """Demonstrate Triton kernel with shared memory"""
    print("\n=== Shared Memory Kernel ===")
    
    @triton.jit
    def shared_memory_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data into shared memory
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Allocate shared memory
        shared_data = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Copy to shared memory (simplified example)
        shared_data = x
        
        # Process in shared memory
        result = shared_data * 2.0
        
        # Store result
        tl.store(out_ptr + offsets, result, mask=mask)
    
    # Test the kernel
    n_elements = 8192
    x = torch.randn(n_elements, device='cuda')
    out = torch.empty_like(x)
    
    threads_per_block = 1024
    blocks_per_grid = triton.cdiv(n_elements, threads_per_block)
    
    shared_memory_kernel[blocks_per_grid](
        x, out, n_elements, BLOCK_SIZE=threads_per_block
    )
    
    expected = x * 2.0
    if torch.allclose(out, expected):
        print("✓ Shared memory kernel works correctly")
    else:
        print("✗ Shared memory kernel failed")

def demonstrate_triton_op_registration():
    """Demonstrate registering Triton kernel as PyTorch op"""
    print("\n=== Triton Op Registration ===")
    
    # Triton compute kernel
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        start = pid * BLOCK_SIZE
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    
    # Register as a Triton-backed PyTorch op
    @triton_op("my_triton_lib::vector_add", mutates_args=())
    def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.device.type == "cuda" and y.device.type == "cuda"
        n = x.numel()
        out = torch.empty_like(x)
        
        # Compute grid size
        def grid_fn(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        
        # Wrap and launch the Triton kernel
        wrap_triton(vector_add_kernel)[grid_fn](x, y, out, n, BLOCK_SIZE=1024)
        return out
    
    # Test the registered op
    a = torch.randn(10000, device='cuda')
    b = torch.randn(10000, device='cuda')
    
    try:
        c = torch.ops.my_triton_lib.vector_add(a, b)
        expected = a + b
        if torch.allclose(c, expected):
            print("✓ Triton op registration works correctly")
        else:
            print("✗ Triton op registration failed")
    except Exception as e:
        print(f"✗ Triton op registration failed: {e}")

def demonstrate_autotuning():
    """Demonstrate Triton autotuning"""
    print("\n=== Triton Autotuning ===")
    
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        ],
        key=['n_elements']
    )
    @triton.jit
    def autotuned_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        x = tl.load(x_ptr + offsets, mask=mask)
        result = x * 2.0
        tl.store(out_ptr + offsets, result, mask=mask)
    
    # Test autotuning with different sizes
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        x = torch.randn(size, device='cuda')
        out = torch.empty_like(x)
        
        # Launch kernel (autotuning will happen on first run)
        blocks_per_grid = triton.cdiv(size, 1024)  # Use max block size for grid
        autotuned_kernel[blocks_per_grid](x, out, size, BLOCK_SIZE=1024)
        
        expected = x * 2.0
        if torch.allclose(out, expected):
            print(f"✓ Autotuned kernel works for size {size}")
        else:
            print(f"✗ Autotuned kernel failed for size {size}")

def demonstrate_persistent_kernel():
    """Demonstrate persistent kernel for matrix multiplication"""
    print("\n=== Persistent Kernel ===")
    
    @triton.jit
    def persistent_gemm_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Compute starting offsets for this block
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N
        
        # Allocate shared memory buffers for A and B tiles
        A_sh = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        B_sh = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        
        # Accumulator for C block
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Loop over K dimension in chunks of BLOCK_K
        num_tiles = (K + BLOCK_K - 1) // BLOCK_K
        for t in range(num_tiles):
            k0 = t * BLOCK_K
            
            # Load A and B tiles into shared memory
            a_ptrs = A_ptr + (offs_am + tl.arange(0, BLOCK_M))[:, None] * stride_am + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_ak
            b_ptrs = B_ptr + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_bk + (offs_bn + tl.arange(0, BLOCK_N))[None, :] * stride_bn
            
            A_sh = tl.load(a_ptrs)
            B_sh = tl.load(b_ptrs)
            
            # Compute partial matmul for this tile
            acc += tl.dot(A_sh, B_sh)
        
        # Write back result
        c_ptrs = C_ptr + (offs_am + tl.arange(0, BLOCK_M))[:, None] * stride_cm + (offs_bn + tl.arange(0, BLOCK_N))[None, :] * stride_cn
        tl.store(c_ptrs, acc)
    
    def persistent_matmul(A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        # Define block sizes
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        
        # Define grid size
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        
        # Launch Triton kernel
        persistent_gemm_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )
        
        return C
    
    # Test persistent kernel
    A = torch.randn(256, 256, device='cuda')
    B = torch.randn(256, 256, device='cuda')
    
    try:
        C = persistent_matmul(A, B)
        expected = torch.mm(A, B)
        if torch.allclose(C, expected, atol=1e-3):
            print("✓ Persistent kernel works correctly")
        else:
            print("✗ Persistent kernel failed")
    except Exception as e:
        print(f"✗ Persistent kernel failed: {e}")

def demonstrate_pipelined_kernel():
    """Demonstrate pipelined kernel with double-buffering"""
    print("\n=== Pipelined Kernel ===")
    
    @triton.jit
    def pipelined_matmul(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        NUM_STAGES: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # Pipelined loop: Triton will issue async copies and overlap with compute
        for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES):
            # Load A and B tiles into shared memory
            a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + tl.arange(0, BLOCK_K))[None, :] * stride_ak)
            b_ptrs = B_ptr + ((k + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn)
            
            A_sh = tl.load(a_ptrs)
            B_sh = tl.load(b_ptrs)
            
            # Compute partial dot
            acc += tl.dot(A_sh, B_sh)
        
        # Write C
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        tl.store(c_ptrs, acc)
    
    def pipelined_matmul_wrapper(A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        NUM_STAGES = 2
        
        grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
        
        pipelined_matmul[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_STAGES=NUM_STAGES
        )
        
        return C
    
    # Test pipelined kernel
    A = torch.randn(128, 128, device='cuda')
    B = torch.randn(128, 128, device='cuda')
    
    try:
        C = pipelined_matmul_wrapper(A, B)
        expected = torch.mm(A, B)
        if torch.allclose(C, expected, atol=1e-3):
            print("✓ Pipelined kernel works correctly")
        else:
            print("✗ Pipelined kernel failed")
    except Exception as e:
        print(f"✗ Pipelined kernel failed: {e}")

def demonstrate_warp_specialization():
    """Demonstrate warp specialization"""
    print("\n=== Warp Specialization ===")
    
    @triton.jit
    def warp_specialized_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Use warp specialization for the loop
        for i in tl.range(0, BLOCK_SIZE, num_stages=2, warp_specialize=True):
            idx = block_start + i
            if idx < n_elements:
                x = tl.load(x_ptr + idx)
                result = x * 2.0
                tl.store(out_ptr + idx, result)
    
    # Test warp specialization
    n_elements = 8192
    x = torch.randn(n_elements, device='cuda')
    out = torch.empty_like(x)
    
    threads_per_block = 1024
    blocks_per_grid = triton.cdiv(n_elements, threads_per_block)
    
    try:
        warp_specialized_kernel[blocks_per_grid](
            x, out, n_elements, BLOCK_SIZE=threads_per_block
        )
        
        expected = x * 2.0
        if torch.allclose(out, expected):
            print("✓ Warp specialization kernel works correctly")
        else:
            print("✗ Warp specialization kernel failed")
    except Exception as e:
        print(f"✗ Warp specialization kernel failed: {e}")

def demonstrate_tensor_cores():
    """Demonstrate Tensor Core usage with WMMA"""
    print("\n=== Tensor Cores ===")
    
    @triton.jit
    def wmma_kernel(
        A_ptr, B_ptr, C_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Each warp handles one 16x16x16 MMA tile
        warp_id = tl.program_id(0)
        lane = tl.arange(0, 32)  # 32 threads in a warp
        
        # Simplified WMMA implementation
        # Note: This is a simplified example - real WMMA requires careful PTX
        a = tl.load(A_ptr + warp_id * BLOCK_M * stride_am + (tl.arange(0, BLOCK_M)[:, None] * stride_am))
        b = tl.load(B_ptr + warp_id * BLOCK_N * stride_bn + (tl.arange(0, BLOCK_K)[:, None] * stride_bk))
        
        # Use tl.dot which can automatically use Tensor Cores for FP16/BF16
        c = tl.dot(a, b)
        
        # Store the result tile
        tl.store(C_ptr + warp_id * BLOCK_M * stride_cm + (tl.arange(0, BLOCK_M)[:, None] * stride_cm), c)
    
    def test_wmma():
        # Use FP16 for Tensor Cores
        A = torch.randn(16, 16, device='cuda', dtype=torch.float16)
        B = torch.randn(16, 16, device='cuda', dtype=torch.float16)
        C = torch.empty_like(A)
        
        grid = (1,)  # one warp (one 16x16 tile)
        
        try:
            wmma_kernel[grid](
                A, B, C,
                16, 16, 16,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=16, BLOCK_N=16, BLOCK_K=16
            )
            
            expected = torch.mm(A.float(), B.float()).half()
            if torch.allclose(C, expected, atol=1e-2):
                print("✓ Tensor Core kernel works correctly")
            else:
                print("✗ Tensor Core kernel failed")
        except Exception as e:
            print(f"✗ Tensor Core kernel failed: {e}")
    
    test_wmma()

def main():
    """Main function demonstrating all Triton techniques"""
    print("Chapter 14: OpenAI Triton Kernel Development")
    print("=" * 60)
    
    # Run all demonstrations
    demonstrate_basic_triton_kernel()
    demonstrate_shared_memory_kernel()
    demonstrate_triton_op_registration()
    demonstrate_autotuning()
    demonstrate_persistent_kernel()
    demonstrate_pipelined_kernel()
    demonstrate_warp_specialization()
    demonstrate_tensor_cores()
    
    print("\n" + "=" * 60)
    print("Triton kernel development completed!")
    print("\nKey takeaways:")
    print("- Use @triton.jit for kernel compilation")
    print("- Use @triton.autotune for automatic optimization")
    print("- Register kernels with torch.library.triton_op")
    print("- Use shared memory for better performance")
    print("- Implement pipelining for memory bandwidth")
    print("- Use warp specialization for complex loops")
    print("- Leverage Tensor Cores with tl.dot")

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
