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

import os
import time
import subprocess
import argparse
import torch
import torch.nn as nn
import torch.profiler as profiler
import numpy as np
import json
from pathlib import Path
import tempfile
import shutil

# CUDA imports
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("Warning: cupy not available")

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: triton not available")

try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Warning: transformer_engine not available")

class KernelEfficiencyProfiler:
    """Comprehensive kernel efficiency and arithmetic intensity profiling with multiple tools"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiling_results = {}
        
    def run_nsys_profile(self, command, output_file="kernel_efficiency_profile"):
        """Run NVIDIA Nsight Systems profiling"""
        print(f"Running Nsight Systems profiling: {command}")
        
        nsys_cmd = [
            "nsys", "profile", 
            "--stats=true",
            "--force-overwrite=true",
            f"--output={output_file}",
            "python", "-c", command
        ]
        
        try:
            result = subprocess.run(nsys_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Nsight Systems profiling completed")
                return result.stdout
            else:
                print(f"✗ Nsight Systems profiling failed: {result.stderr}")
                return None
        except FileNotFoundError:
            print("✗ nsys not found. Install NVIDIA Nsight Systems")
            return None
    
    def run_ncu_profile(self, command, output_file="kernel_efficiency_ncu"):
        """Run NVIDIA Nsight Compute profiling"""
        print(f"Running Nsight Compute profiling: {command}")
        
        ncu_cmd = [
            "ncu", 
            "--set", "full",
            "-o", output_file,
            "python", "-c", command
        ]
        
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Nsight Compute profiling completed")
                return result.stdout
            else:
                print(f"✗ Nsight Compute profiling failed: {result.stderr}")
                return None
        except FileNotFoundError:
            print("✗ ncu not found. Install NVIDIA Nsight Compute")
            return None
    
    def run_pytorch_profiler(self, model, dataloader, num_batches=10):
        """Run PyTorch profiler with detailed kernel efficiency analysis"""
        print("Running PyTorch profiler with kernel efficiency analysis...")
        
        model.eval()
        
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(
                wait=1,
                warmup=1,
                active=num_batches,
                repeat=1
            ),
            on_trace_ready=profiler.tensorboard_trace_handler('./log/kernel_efficiency_profiler'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for i, (data, target) in enumerate(dataloader):
                if i >= num_batches:
                    break
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                with torch.no_grad():
                    output = model(data)
                    loss = nn.functional.cross_entropy(output, target)
                
                if i == 0:  # Warmup
                    continue
                    
        print("✓ PyTorch profiler completed")
        return prof
    
    def run_memory_profiler(self, func, *args, **kwargs):
        """Run PyTorch memory profiler for kernel efficiency analysis"""
        print("Running PyTorch memory profiler...")
        
        # Clear cache before profiling
        torch.cuda.empty_cache()
        
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True
        ) as prof:
            result = func(*args, **kwargs)
        
        print("✓ Memory profiler completed")
        return prof, result
    
    def run_hta_profile(self, command):
        """Run HTA (Hierarchical Timeline Analysis) profiling"""
        print(f"Running HTA profiling: {command}")
        
        hta_cmd = [
            "python", "-c", 
            f"""
import torch
import torch.profiler as profiler
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from code.ch9.kernel_efficiency_arithmetic_intensity import SimpleModel, MockDataset

# Setup
dataset = MockDataset(100)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
model = SimpleModel().cuda()

# HTA-style profiling
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler('./log/hta_kernel_efficiency_profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i, (data, target) in enumerate(dataloader):
        if i >= 5:
            break
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        with torch.no_grad():
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            
print("HTA kernel efficiency profiling completed")
            """
        ]
        
        try:
            result = subprocess.run(hta_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ HTA profiling completed")
                return result.stdout
            else:
                print(f"✗ HTA profiling failed: {result.stderr}")
                return None
        except Exception as e:
            print(f"✗ HTA profiling error: {e}")
            return None
    
    def run_perf_profile(self, command):
        """Run Linux perf profiling for system-level kernel efficiency analysis"""
        print(f"Running perf profiling: {command}")
        
        perf_cmd = [
            "perf", "record", 
            "-g",  # Call graph
            "-F", "99",  # 99 Hz sampling
            "-o", "perf_kernel_efficiency.data",
            "python", "-c", command
        ]
        
        try:
            result = subprocess.run(perf_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Generate report
                report_cmd = ["perf", "report", "-i", "perf_kernel_efficiency.data"]
                report_result = subprocess.run(report_cmd, capture_output=True, text=True)
                print("✓ perf profiling completed")
                return report_result.stdout
            else:
                print(f"✗ perf profiling failed: {result.stderr}")
                return None
        except FileNotFoundError:
            print("✗ perf not found. Install linux-tools-common")
            return None

class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for kernel efficiency testing"""
    def __init__(self, size=1000, input_size=224*224*3):
        self.size = size
        self.input_size = input_size
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, 10, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SimpleModel(nn.Module):
    """Simple model for kernel efficiency testing"""
    def __init__(self, input_size=224*224*3, hidden_size=512, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_arithmetic_intensity():
    """Test arithmetic intensity optimization"""
    print("Testing Arithmetic Intensity:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Low arithmetic intensity (memory-bound)
    print("\n1. Low Arithmetic Intensity (Memory-Bound)")
    size = 10_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c = a + b  # Simple addition - low arithmetic intensity
    torch.cuda.synchronize()
    low_intensity_time = time.time() - start_time
    
    bytes_moved = size * 4 * 3  # 3 tensors, 4 bytes each
    flops = size  # 1 FLOP per element
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  Low intensity time: {low_intensity_time*1000:.2f} ms")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.3f} FLOPs/byte")
    
    # Test 2: High arithmetic intensity (compute-bound)
    print("\n2. High Arithmetic Intensity (Compute-Bound)")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Heavy computation with high arithmetic intensity
    for i in range(100):
        a = torch.sin(a) + torch.cos(a) + torch.tan(a) + torch.exp(a)
    
    torch.cuda.synchronize()
    high_intensity_time = time.time() - start_time
    
    flops = size * 100 * 4  # 4 FLOPs per element per iteration
    bytes_moved = size * 4 * 2  # Rough estimate
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  High intensity time: {high_intensity_time*1000:.2f} ms")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")
    
    # Test 3: Balanced arithmetic intensity
    print("\n3. Balanced Arithmetic Intensity")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Balanced computation
    for i in range(10):
        c = a * b + a + b
        a = c * 2.0 + torch.sin(c)
    
    torch.cuda.synchronize()
    balanced_time = time.time() - start_time
    
    flops = size * 10 * 5  # 5 FLOPs per element per iteration
    bytes_moved = size * 4 * 3 * 10  # Rough estimate
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  Balanced time: {balanced_time*1000:.2f} ms")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")

def test_kernel_fusion():
    """Test kernel fusion for improved arithmetic intensity"""
    print("\nTesting Kernel Fusion:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Unfused kernels (low efficiency)
    print("\n1. Unfused Kernels (Low Efficiency)")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Multiple separate operations
    b = torch.sin(a)
    c = torch.sqrt(b)
    d = torch.exp(c)
    e = d * 2.0
    
    torch.cuda.synchronize()
    unfused_time = time.time() - start_time
    
    print(f"  Unfused time: {unfused_time*1000:.2f} ms")
    
    # Test 2: Fused kernel (high efficiency)
    print("\n2. Fused Kernel (High Efficiency)")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Single fused operation
    e_fused = torch.exp(torch.sqrt(torch.sin(a))) * 2.0
    
    torch.cuda.synchronize()
    fused_time = time.time() - start_time
    
    print(f"  Fused time: {fused_time*1000:.2f} ms")
    print(f"  Fusion speedup: {unfused_time/fused_time:.1f}x")
    
    # Verify results
    assert torch.allclose(e, e_fused, atol=1e-5)
    print("  ✓ Results verified")
    
    # Test 3: Complex fusion with multiple operations
    print("\n3. Complex Fusion")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Complex unfused operations
    c1 = a + b
    c2 = torch.sin(c1)
    c3 = torch.cos(c1)
    c4 = c2 * c3
    c5 = torch.sqrt(c4)
    c6 = torch.exp(c5)
    
    torch.cuda.synchronize()
    complex_unfused_time = time.time() - start_time
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Complex fused operation
    c6_fused = torch.exp(torch.sqrt(torch.sin(a + b) * torch.cos(a + b)))
    
    torch.cuda.synchronize()
    complex_fused_time = time.time() - start_time
    
    print(f"  Complex unfused time: {complex_unfused_time*1000:.2f} ms")
    print(f"  Complex fused time: {complex_fused_time*1000:.2f} ms")
    print(f"  Complex fusion speedup: {complex_unfused_time/complex_fused_time:.1f}x")
    
    # Verify results
    assert torch.allclose(c6, c6_fused, atol=1e-5)
    print("  ✓ Complex results verified")

def test_tiling_optimization():
    """Test tiling optimization for improved arithmetic intensity"""
    print("\nTesting Tiling Optimization:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Naive matrix multiplication (no tiling)
    print("\n1. Naive Matrix Multiplication (No Tiling)")
    m, n, k = 1024, 1024, 1024
    
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c_naive = torch.mm(a, b)
    torch.cuda.synchronize()
    naive_time = time.time() - start_time
    
    flops = 2 * m * n * k
    gflops = flops / naive_time / 1e9
    print(f"  Naive time: {naive_time*1000:.2f} ms")
    print(f"  Naive performance: {gflops:.1f} GFLOPS")
    
    # Test 2: Tiled matrix multiplication (simulated)
    print("\n2. Tiled Matrix Multiplication")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate tiling with smaller chunks
    tile_size = 64
    c_tiled = torch.zeros(m, n, device=device)
    
    for i in range(0, m, tile_size):
        for j in range(0, n, tile_size):
            for k_idx in range(0, k, tile_size):
                end_i = min(i + tile_size, m)
                end_j = min(j + tile_size, n)
                end_k = min(k_idx + tile_size, k)
                
                c_tiled[i:end_i, j:end_j] += torch.mm(
                    a[i:end_i, k_idx:end_k], 
                    b[k_idx:end_k, j:end_j]
                )
    
    torch.cuda.synchronize()
    tiled_time = time.time() - start_time
    
    gflops = flops / tiled_time / 1e9
    print(f"  Tiled time: {tiled_time*1000:.2f} ms")
    print(f"  Tiled performance: {gflops:.1f} GFLOPS")
    print(f"  Tiling speedup: {naive_time/tiled_time:.1f}x")
    
    # Verify results
    assert torch.allclose(c_naive, c_tiled, atol=1e-5)
    print("  ✓ Results verified")
    
    # Test 3: Multi-level tiling simulation
    print("\n3. Multi-Level Tiling Simulation")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate multi-level tiling with register-level operations
    chunk_size = 1024  # Shared memory level
    sub_chunk_size = 256  # Register level
    
    result = torch.zeros_like(a)
    for i in range(0, size, chunk_size):
        end_chunk = min(i + chunk_size, size)
        chunk_data = a[i:end_chunk]
        
        # Process in smaller sub-chunks (register level)
        for j in range(0, len(chunk_data), sub_chunk_size):
            end_sub = min(j + sub_chunk_size, len(chunk_data))
            sub_data = chunk_data[j:end_sub]
            
            # Heavy computation on sub-chunk
            result[i+j:i+end_sub] = torch.sin(sub_data) + torch.cos(sub_data) + torch.tan(sub_data)
    
    torch.cuda.synchronize()
    multi_level_time = time.time() - start_time
    
    print(f"  Multi-level tiled time: {multi_level_time*1000:.2f} ms")

def test_precision_optimization():
    """Test precision optimization for improved arithmetic intensity"""
    print("\nTesting Precision Optimization:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: FP32 precision
    print("\n1. FP32 Precision")
    size = 1_000_000
    
    a = torch.randn(size, dtype=torch.float32, device=device)
    b = torch.randn(size, dtype=torch.float32, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c_fp32 = torch.mm(a.view(-1, 1000), b.view(1000, -1))
    torch.cuda.synchronize()
    fp32_time = time.time() - start_time
    
    print(f"  FP32 time: {fp32_time*1000:.2f} ms")
    
    # Test 2: FP16 precision
    print("\n2. FP16 Precision")
    a_half = torch.randn(size, dtype=torch.float16, device=device)
    b_half = torch.randn(size, dtype=torch.float16, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c_fp16 = torch.mm(a_half.view(-1, 1000), b_half.view(1000, -1))
    torch.cuda.synchronize()
    fp16_time = time.time() - start_time
    
    print(f"  FP16 time: {fp16_time*1000:.2f} ms")
    print(f"  FP16 speedup: {fp32_time/fp16_time:.1f}x")
    
    # Test 3: Mixed precision
    print("\n3. Mixed Precision")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Mixed precision computation
    with torch.cuda.amp.autocast():
        c_mixed = torch.mm(a.view(-1, 1000), b.view(1000, -1))
    
    torch.cuda.synchronize()
    mixed_time = time.time() - start_time
    
    print(f"  Mixed precision time: {mixed_time*1000:.2f} ms")
    print(f"  Mixed precision speedup: {fp32_time/mixed_time:.1f}x")

def test_structured_sparsity():
    """Test structured sparsity optimization"""
    print("\nTesting Structured Sparsity:")
    print("=" * 50)
    
    if not TE_AVAILABLE:
        print("Transformer Engine not available")
        return
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Dense matrix multiplication
    print("\n1. Dense Matrix Multiplication")
    m, n, k = 1024, 1024, 1024
    
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c_dense = torch.mm(a, b)
    torch.cuda.synchronize()
    dense_time = time.time() - start_time
    
    flops = 2 * m * n * k
    gflops = flops / dense_time / 1e9
    print(f"  Dense time: {dense_time*1000:.2f} ms")
    print(f"  Dense performance: {gflops:.1f} GFLOPS")
    
    # Test 2: Structured sparsity (2:4 pattern)
    print("\n2. Structured Sparsity (2:4 Pattern)")
    
    # Create 2:4 sparse pattern
    def create_2_4_sparse(tensor):
        # Create 2:4 sparsity pattern (2 non-zeros per 4 elements)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        for i in range(0, tensor.numel(), 4):
            if i + 3 < tensor.numel():
                # Set 2 out of 4 elements to True
                indices = torch.randperm(4)[:2]
                mask.view(-1)[i:i+4][indices] = True
        return tensor * mask
    
    a_sparse = create_2_4_sparse(a)
    b_sparse = create_2_4_sparse(b)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c_sparse = torch.mm(a_sparse, b_sparse)
    torch.cuda.synchronize()
    sparse_time = time.time() - start_time
    
    gflops = flops / sparse_time / 1e9
    print(f"  Sparse time: {sparse_time*1000:.2f} ms")
    print(f"  Sparse performance: {gflops:.1f} GFLOPS")
    print(f"  Sparsity speedup: {dense_time/sparse_time:.1f}x")
    
    # Test 3: Transformer Engine sparse operations
    print("\n3. Transformer Engine Sparse Operations")
    
    # Create sparse linear layer
    sparse_layer = te.Linear(1024, 1024, bias=False)
    
    # Convert to sparse format
    try:
        sparse_layer = te.Linear.from_pytorch_linear(sparse_layer)
        sparse_layer = sparse_layer.to_sparse_semi_structured()
        
        input_tensor = torch.randn(32, 1024, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        output_sparse = sparse_layer(input_tensor)
        torch.cuda.synchronize()
        te_sparse_time = time.time() - start_time
        
        print(f"  TE sparse time: {te_sparse_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"  TE sparse operation failed: {e}")

def test_triton_kernel_efficiency():
    """Test Triton-based kernel efficiency optimization"""
    print("\nTesting Triton Kernel Efficiency:")
    print("=" * 50)
    
    if not TRITON_AVAILABLE:
        print("Triton not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Basic Triton kernel with high arithmetic intensity
    print("\n1. Basic Triton Kernel with High Arithmetic Intensity")
    
    @triton.jit
    def high_intensity_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # High arithmetic intensity operations
        y = tl.sin(x) + tl.cos(x) + tl.tan(x) + tl.exp(x)
        y = y * 2.0 + tl.sqrt(tl.abs(y))
        
        # Store result
        tl.store(y_ptr + offsets, y, mask=mask)
    
    size = 1_000_000
    x = torch.randn(size, device=device)
    y = torch.empty_like(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    high_intensity_kernel[grid](x, y, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  High intensity Triton kernel time: {elapsed*1000:.2f} ms")
    
    # Verify result
    expected = torch.sin(x) + torch.cos(x) + torch.tan(x) + torch.exp(x)
    expected = expected * 2.0 + torch.sqrt(torch.abs(expected))
    assert torch.allclose(y, expected, atol=1e-5)
    print("  ✓ Result verified")
    
    # Test 2: Fused Triton kernel
    print("\n2. Fused Triton Kernel")
    
    @triton.jit
    def fused_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load both inputs
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        
        # Fused operations
        c = tl.sin(a + b) * tl.cos(a - b) + tl.exp(a * b)
        
        # Store result
        tl.store(c_ptr + offsets, c, mask=mask)
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    c = torch.empty_like(a)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    fused_kernel[grid](a, b, c, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  Fused Triton kernel time: {elapsed*1000:.2f} ms")
    
    # Verify result
    expected = torch.sin(a + b) * torch.cos(a - b) + torch.exp(a * b)
    assert torch.allclose(c, expected, atol=1e-5)
    print("  ✓ Fused result verified")

def test_kernel_efficiency_profiling():
    """Test comprehensive kernel efficiency profiling"""
    print("\nTesting Kernel Efficiency Profiling:")
    print("=" * 50)
    
    profiler = KernelEfficiencyProfiler()
    
    # Test 1: Basic kernel efficiency profiling
    print("\n1. Basic Kernel Efficiency Profiling")
    basic_command = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from code.ch9.kernel_efficiency_arithmetic_intensity import MockDataset, SimpleModel

dataset = MockDataset(100)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
model = SimpleModel().cuda()

for i, (data, target) in enumerate(dataloader):
    if i >= 5:
        break
    data = data.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    
    with torch.no_grad():
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
"""
    
    # Run different profiling tools
    profiler.run_nsys_profile(basic_command)
    profiler.run_ncu_profile(basic_command)
    profiler.run_hta_profile(basic_command)
    profiler.run_perf_profile(basic_command)
    
    # Test 2: PyTorch profiler with kernel efficiency analysis
    print("\n2. PyTorch Profiler with Kernel Efficiency Analysis")
    dataset = MockDataset(100)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    model = SimpleModel().cuda()
    
    pytorch_prof = profiler.run_pytorch_profiler(model, dataloader, num_batches=5)
    
    # Test 3: Memory profiling
    print("\n3. Kernel Efficiency Memory Profiling")
    def memory_test_func():
        dataset = MockDataset(100)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
        model = SimpleModel().cuda()
        
        for i, (data, target) in enumerate(dataloader):
            if i >= 5:
                break
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            with torch.no_grad():
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
        
        return model
    
    memory_prof, result = profiler.run_memory_profiler(memory_test_func)
    
    return {
        'pytorch_prof': pytorch_prof,
        'memory_prof': memory_prof
    }

def test_roofline_analysis():
    """Test roofline analysis for kernel efficiency"""
    print("\nTesting Roofline Analysis:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Memory-bound kernel
    print("\n1. Memory-Bound Kernel")
    size = 10_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c = a + b  # Simple memory-bound operation
    torch.cuda.synchronize()
    memory_time = time.time() - start_time
    
    bytes_moved = size * 4 * 3
    memory_bandwidth = bytes_moved / memory_time / (1024**3)
    flops = size
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  Memory-bound time: {memory_time*1000:.2f} ms")
    print(f"  Memory bandwidth: {memory_bandwidth:.1f} GB/s")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.3f} FLOPs/byte")
    
    # Test 2: Compute-bound kernel
    print("\n2. Compute-Bound Kernel")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Heavy computation
    for i in range(100):
        a = torch.sin(a) + torch.cos(a) + torch.tan(a) + torch.exp(a)
    
    torch.cuda.synchronize()
    compute_time = time.time() - start_time
    
    flops = size * 100 * 4
    bytes_moved = size * 4 * 2
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  Compute-bound time: {compute_time*1000:.2f} ms")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")
    
    # Test 3: Balanced kernel
    print("\n3. Balanced Kernel")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Balanced computation
    for i in range(10):
        c = a * b + a + b
        a = c * 2.0 + torch.sin(c)
    
    torch.cuda.synchronize()
    balanced_time = time.time() - start_time
    
    flops = size * 10 * 4
    bytes_moved = size * 4 * 3 * 10
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  Balanced time: {balanced_time*1000:.2f} ms")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")

def main():
    """Main function demonstrating kernel efficiency and arithmetic intensity optimization"""
    parser = argparse.ArgumentParser(description="Kernel Efficiency and Arithmetic Intensity Examples")
    parser.add_argument("--test", choices=["all", "intensity", "fusion", "tiling", "precision", "sparsity", "triton", "profiling", "roofline"], 
                       default="all", help="Test to run")
    args = parser.parse_args()
    
    print("AI Performance Engineering - Chapter 9")
    print("Kernel Efficiency and Arithmetic Intensity with Comprehensive Profiling")
    print("=" * 70)
    
    if args.test == "intensity" or args.test == "all":
        test_arithmetic_intensity()
    
    if args.test == "fusion" or args.test == "all":
        test_kernel_fusion()
    
    if args.test == "tiling" or args.test == "all":
        test_tiling_optimization()
    
    if args.test == "precision" or args.test == "all":
        test_precision_optimization()
    
    if args.test == "sparsity" or args.test == "all":
        test_structured_sparsity()
    
    if args.test == "triton" or args.test == "all":
        test_triton_kernel_efficiency()
    
    if args.test == "profiling" or args.test == "all":
        test_kernel_efficiency_profiling()
    
    if args.test == "roofline" or args.test == "all":
        test_roofline_analysis()
    
    print("\nKernel Efficiency and Arithmetic Intensity Summary:")
    print("=" * 50)
    print("✓ Increase arithmetic intensity for better compute utilization")
    print("✓ Use kernel fusion to reduce memory traffic")
    print("✓ Implement tiling for data reuse and cache efficiency")
    print("✓ Profile with nsys, ncu, PyTorch profiler, and memory profiler")
    print("✓ Use Triton for custom high-intensity kernels")
    print("✓ Leverage structured sparsity and precision optimization")
    print("✓ Monitor roofline analysis for performance bottlenecks")

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
