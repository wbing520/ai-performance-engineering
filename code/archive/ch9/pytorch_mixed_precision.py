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
# pytorch_mixed_precision.py
# Chapter 9: PyTorch mixed precision and Tensor Core utilization examples

import torch
import time
import numpy as np

def benchmark_matmul(A, B, name, warmup_iters=10, benchmark_iters=100):
    """Benchmark matrix multiplication with timing."""
    # Warm up
    for _ in range(warmup_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(benchmark_iters):
        C = torch.matmul(A, B)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / benchmark_iters
    
    # Calculate FLOPS
    M, K = A.shape
    K2, N = B.shape
    flops = 2 * M * N * K  # Multiply-accumulate operations
    gflops = flops / (elapsed_time * 1e6)
    
    print(f"{name}:")
    print(f"  Time: {elapsed_time:.4f} ms")
    print(f"  Performance: {gflops:.1f} GFLOP/s")
    
    return elapsed_time, gflops, C

def main():
    print("PyTorch Mixed Precision Examples (Chapter 9)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Matrix dimensions for benchmark
    M, N, K = 2048, 2048, 2048
    print(f"\nMatrix dimensions: {M}x{K} @ {K}x{N}")
    
    # Create input matrices
    A_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    B_fp32 = torch.randn(K, N, device=device, dtype=torch.float32)
    
    A_fp16 = A_fp32.half()
    B_fp16 = B_fp32.half()
    
    A_bf16 = A_fp32.to(torch.bfloat16)
    B_bf16 = B_fp32.to(torch.bfloat16)
    
    print("\n" + "="*60)
    print("1. Standard Precision Benchmarks")
    print("="*60)
    
    # FP32 benchmark
    time_fp32, gflops_fp32, C_fp32 = benchmark_matmul(A_fp32, B_fp32, "FP32 MatMul")
    
    # FP16 benchmark
    time_fp16, gflops_fp16, C_fp16 = benchmark_matmul(A_fp16, B_fp16, "FP16 MatMul")
    
    # BF16 benchmark
    time_bf16, gflops_bf16, C_bf16 = benchmark_matmul(A_bf16, B_bf16, "BF16 MatMul")
    
    print(f"\nSpeedups:")
    print(f"  FP16 vs FP32: {time_fp32 / time_fp16:.2f}x")
    print(f"  BF16 vs FP32: {time_fp32 / time_bf16:.2f}x")
    
    print("\n" + "="*60)
    print("2. TF32 Mode (PyTorch 2.8)")
    print("="*60)
    
    # Enable TF32 for even better Tensor Core utilization
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    time_tf32, gflops_tf32, C_tf32 = benchmark_matmul(A_fp32, B_fp32, "TF32 MatMul (FP32 input)")
    
    print(f"\nTF32 speedup vs standard FP32: {time_fp32 / time_tf32:.2f}x")
    
    print("\n" + "="*60)
    print("3. Automatic Mixed Precision (AMP)")
    print("="*60)
    
    def amp_matmul_bf16(A, B):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return torch.matmul(A, B)
    
    def amp_matmul_fp16(A, B):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return torch.matmul(A, B)
    
    # AMP BF16 benchmark
    # Warm up
    for _ in range(10):
        C = amp_matmul_bf16(A_fp32, B_fp32)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        C = amp_matmul_bf16(A_fp32, B_fp32)
    end_event.record()
    
    torch.cuda.synchronize()
    time_amp_bf16 = start_event.elapsed_time(end_event) / 100
    
    # AMP FP16 benchmark
    for _ in range(10):
        C = amp_matmul_fp16(A_fp32, B_fp32)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(100):
        C = amp_matmul_fp16(A_fp32, B_fp32)
    end_event.record()
    
    torch.cuda.synchronize()
    time_amp_fp16 = start_event.elapsed_time(end_event) / 100
    
    flops = 2 * M * N * K
    gflops_amp_bf16 = flops / (time_amp_bf16 * 1e6)
    gflops_amp_fp16 = flops / (time_amp_fp16 * 1e6)
    
    print(f"AMP BF16:")
    print(f"  Time: {time_amp_bf16:.4f} ms")
    print(f"  Performance: {gflops_amp_bf16:.1f} GFLOP/s")
    
    print(f"AMP FP16:")
    print(f"  Time: {time_amp_fp16:.4f} ms")
    print(f"  Performance: {gflops_amp_fp16:.1f} GFLOP/s")
    
    print("\n" + "="*60)
    print("4. torch.compile with Mixed Precision")
    print("="*60)
    
    @torch.compile(fullgraph=True)
    def compiled_matmul(A, B):
        return torch.matmul(A, B)
    
    @torch.compile(fullgraph=True)
    def compiled_amp_matmul(A, B):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return torch.matmul(A, B)
    
    # Compile the functions
    _ = compiled_matmul(A_fp32, B_fp32)
    _ = compiled_amp_matmul(A_fp32, B_fp32)
    torch.cuda.synchronize()
    
    time_compiled, gflops_compiled, _ = benchmark_matmul(A_fp32, B_fp32, "Compiled FP32", 
                                                        benchmark_iters=100)
    
    # Benchmark compiled AMP
    for _ in range(10):
        C = compiled_amp_matmul(A_fp32, B_fp32)
    torch.cuda.synchronize()
    
    start_event.record()
    for _ in range(100):
        C = compiled_amp_matmul(A_fp32, B_fp32)
    end_event.record()
    
    torch.cuda.synchronize()
    time_compiled_amp = start_event.elapsed_time(end_event) / 100
    gflops_compiled_amp = flops / (time_compiled_amp * 1e6)
    
    print(f"Compiled AMP BF16:")
    print(f"  Time: {time_compiled_amp:.4f} ms")
    print(f"  Performance: {gflops_compiled_amp:.1f} GFLOP/s")
    
    print("\n" + "="*60)
    print("5. Arithmetic Intensity Analysis")
    print("="*60)
    
    # Calculate memory bandwidth for different precisions
    bytes_fp32 = (M * K + K * N + M * N) * 4  # 4 bytes per FP32
    bytes_fp16 = (M * K + K * N) * 2 + M * N * 4  # FP16 input, FP32 output
    bytes_bf16 = (M * K + K * N) * 2 + M * N * 4  # BF16 input, FP32 output
    
    ai_fp32 = flops / bytes_fp32
    ai_fp16 = flops / bytes_fp16
    ai_bf16 = flops / bytes_bf16
    
    print(f"Arithmetic Intensity:")
    print(f"  FP32: {ai_fp32:.2f} FLOPs/byte")
    print(f"  FP16: {ai_fp16:.2f} FLOPs/byte")
    print(f"  BF16: {ai_bf16:.2f} FLOPs/byte")
    
    print(f"\nMemory Bandwidth Utilization (estimated):")
    gpu_bw = 2000e9  # Approximate bandwidth for modern GPU (2 TB/s)
    
    bw_util_fp32 = (bytes_fp32 / (time_fp32 * 1e-3)) / gpu_bw * 100
    bw_util_fp16 = (bytes_fp16 / (time_fp16 * 1e-3)) / gpu_bw * 100
    bw_util_bf16 = (bytes_bf16 / (time_bf16 * 1e-3)) / gpu_bw * 100
    
    print(f"  FP32: {bw_util_fp32:.1f}%")
    print(f"  FP16: {bw_util_fp16:.1f}%")
    print(f"  BF16: {bw_util_bf16:.1f}%")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Best performance: {max(gflops_fp32, gflops_fp16, gflops_bf16, gflops_compiled_amp):.1f} GFLOP/s")
    print("Key takeaways:")
    print("- Modern Tensor Cores provide significant speedups for FP16/BF16")
    print("- TF32 mode improves FP32 performance on Tensor Cores")
    print("- AMP automatically selects optimal precision per operation")
    print("- torch.compile can further optimize mixed-precision workflows")

if __name__ == "__main__":
    main()
