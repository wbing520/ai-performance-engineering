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
# pytorch_fused_gemm.py
# Chapter 10: PyTorch compiled GEMM demonstrating automatic pipelining

import torch
import time

@torch.compile(fullgraph=True)
def fused_gemm(A, B):
    return torch.matmul(A, B)

def main():
    print("PyTorch Fused GEMM Example (Chapter 10)")
    print(f"PyTorch version: {torch.__version__}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Matrix dimensions
    M, N, K = 2048, 2048, 2048
    print(f"Matrix size: {M}x{N}x{K}")
    
    # Create input tensors
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    
    print("\n" + "="*50)
    print("1. Standard torch.matmul")
    print("="*50)
    
    # Warm up
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark standard matmul
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        C_standard = torch.matmul(A, B)
    end_event.record()
    
    torch.cuda.synchronize()
    
    standard_time = start_event.elapsed_time(end_event) / 100
    flops = 2 * M * N * K
    standard_gflops = flops / (standard_time * 1e6)
    
    print(f"Standard matmul time: {standard_time:.4f} ms")
    print(f"Standard matmul performance: {standard_gflops:.1f} GFLOP/s")
    
    print("\n" + "="*50)
    print("2. torch.compile fused GEMM")
    print("="*50)
    
    # Compile the function (first call triggers compilation)
    print("Compiling function...")
    C_compiled = fused_gemm(A, B)
    torch.cuda.synchronize()
    print("Compilation complete.")
    
    # Warm up compiled version
    for _ in range(10):
        C_compiled = fused_gemm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark compiled version
    start_event.record()
    for _ in range(100):
        C_compiled = fused_gemm(A, B)
    end_event.record()
    
    torch.cuda.synchronize()
    
    compiled_time = start_event.elapsed_time(end_event) / 100
    compiled_gflops = flops / (compiled_time * 1e6)
    
    print(f"Compiled fused GEMM time: {compiled_time:.4f} ms")
    print(f"Compiled fused GEMM performance: {compiled_gflops:.1f} GFLOP/s")
    print(f"Speedup: {standard_time / compiled_time:.2f}x")
    
    print("\n" + "="*50)
    print("3. Mixed precision with compilation")
    print("="*50)
    
    @torch.compile(fullgraph=True)
    def fused_gemm_mixed(A, B):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return torch.matmul(A, B)
    
    # Compile mixed precision version
    C_mixed = fused_gemm_mixed(A, B)
    torch.cuda.synchronize()
    
    # Warm up
    for _ in range(10):
        C_mixed = fused_gemm_mixed(A, B)
    torch.cuda.synchronize()
    
    # Benchmark mixed precision
    start_event.record()
    for _ in range(100):
        C_mixed = fused_gemm_mixed(A, B)
    end_event.record()
    
    torch.cuda.synchronize()
    
    mixed_time = start_event.elapsed_time(end_event) / 100
    mixed_gflops = flops / (mixed_time * 1e6)
    
    print(f"Mixed precision compiled time: {mixed_time:.4f} ms")
    print(f"Mixed precision compiled performance: {mixed_gflops:.1f} GFLOP/s")
    print(f"Speedup over standard: {standard_time / mixed_time:.2f}x")
    
    print("\n" + "="*50)
    print("4. Fused attention example")
    print("="*50)
    
    # Demonstrate fused attention
    batch_size, seq_len, head_dim = 32, 1024, 64
    
    queries = torch.randn(batch_size, seq_len, head_dim, device=device)
    keys = torch.randn(batch_size, seq_len, head_dim, device=device)
    values = torch.randn(batch_size, seq_len, head_dim, device=device)
    
    print(f"Attention dimensions: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
    
    # Manual attention (3 separate operations)
    def manual_attention(q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1))
        probabilities = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, v)
        return context
    
    # Fused attention
    def fused_attention(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
    
    # Warm up both versions
    for _ in range(10):
        manual_result = manual_attention(queries, keys, values)
        fused_result = fused_attention(queries, keys, values)
    torch.cuda.synchronize()
    
    # Benchmark manual attention
    start_event.record()
    for _ in range(100):
        manual_result = manual_attention(queries, keys, values)
    end_event.record()
    torch.cuda.synchronize()
    
    manual_attn_time = start_event.elapsed_time(end_event) / 100
    
    # Benchmark fused attention
    start_event.record()
    for _ in range(100):
        fused_result = fused_attention(queries, keys, values)
    end_event.record()
    torch.cuda.synchronize()
    
    fused_attn_time = start_event.elapsed_time(end_event) / 100
    
    print(f"Manual attention time: {manual_attn_time:.4f} ms")
    print(f"Fused attention time: {fused_attn_time:.4f} ms")
    print(f"Fused attention speedup: {manual_attn_time / fused_attn_time:.2f}x")
    
    print("\n" + "="*50)
    print("Benefits of PyTorch Compilation & Fusion")
    print("="*50)
    
    print("torch.compile benefits:")
    print("- Automatic kernel fusion eliminates intermediate memory traffic")
    print("- Uses CUTLASS and NVFuser for optimized CUDA kernels")
    print("- Automatically applies warp specialization and pipelining")
    print("- Equivalent to <cuda/pipeline> primitives under the hood")
    print("- Producer-consumer handoffs without explicit __syncthreads()")
    
    print("\nFused attention benefits:")
    print("- Combines matmul + softmax + matmul into single kernel")
    print("- Loader, compute, and store warps overlap asynchronously")
    print("- Eliminates expensive global memory round trips")
    print("- Achieves ~75% of peak FLOPS throughput")
    
    # Verify results are close
    print("\n" + "="*50)
    print("Verification")
    print("="*50)
    
    # Check that compiled and standard versions produce similar results
    diff = torch.max(torch.abs(C_standard - C_compiled)).item()
    print(f"Max difference between standard and compiled GEMM: {diff:.2e}")
    
    # Check attention results
    attn_diff = torch.max(torch.abs(manual_result - fused_result)).item()
    print(f"Max difference between manual and fused attention: {attn_diff:.2e}")
    
    tolerance = 1e-4
    gemm_pass = diff < tolerance
    attn_pass = attn_diff < tolerance
    
    print(f"GEMM verification: {'PASS' if gemm_pass else 'FAIL'}")
    print(f"Attention verification: {'PASS' if attn_pass else 'FAIL'}")

if __name__ == "__main__":
    main()
