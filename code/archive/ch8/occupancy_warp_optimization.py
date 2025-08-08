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

class OccupancyProfiler:
    """Comprehensive occupancy and warp efficiency profiling with multiple tools"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiling_results = {}
        
    def run_nsys_profile(self, command, output_file="occupancy_profile"):
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
    
    def run_ncu_profile(self, command, output_file="occupancy_ncu"):
        """Run NVIDIA Nsight Compute profiling"""
        print(f"Running Nsight Compute profiling: {command}")
        
        ncu_cmd = [
            "ncu", 
            "--set", "full",
            "--export", output_file,
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
        """Run PyTorch profiler with detailed occupancy analysis"""
        print("Running PyTorch profiler with occupancy analysis...")
        
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
            on_trace_ready=profiler.tensorboard_trace_handler('./log/occupancy_profiler'),
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
        """Run PyTorch memory profiler for occupancy analysis"""
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
from code.ch8.occupancy_warp_optimization import SimpleModel, MockDataset

# Setup
dataset = MockDataset(100)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
model = SimpleModel().cuda()

# HTA-style profiling
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler('./log/hta_occupancy_profile'),
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
            
print("HTA occupancy profiling completed")
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
        """Run Linux perf profiling for system-level occupancy analysis"""
        print(f"Running perf profiling: {command}")
        
        perf_cmd = [
            "perf", "record", 
            "-g",  # Call graph
            "-F", "99",  # 99 Hz sampling
            "-o", "perf_occupancy.data",
            "python", "-c", command
        ]
        
        try:
            result = subprocess.run(perf_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Generate report
                report_cmd = ["perf", "report", "-i", "perf_occupancy.data"]
                report_result = subprocess.run(report_cmd, capture_output=True, text=True)
                print("✓ perf profiling completed")
                return report_result.stdout
            else:
                print(f"✗ perf profiling failed: {result.stderr}")
                return None
        except FileNotFoundError:
            print("✗ perf not found. Install linux-tools-common")
            return None
    
    def get_gpu_occupancy_info(self):
        """Get detailed GPU occupancy information"""
        print("Getting GPU occupancy information...")
        
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        occupancy_info = {
            'device_name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'max_threads_per_block': props.max_threads_per_block,
            'max_blocks_per_sm': 32,  # Typical for modern GPUs
            'max_warps_per_sm': 64,   # Typical for modern GPUs
            'max_shared_memory_per_block': props.max_shared_memory_per_block,
            'max_shared_memory_per_sm': 228 * 1024,  # 228 KB for Blackwell
            'warp_size': props.warp_size,
            'max_grid_size': props.max_grid_size,
            'max_block_size': props.max_block_size,
            'max_registers_per_block': 65536,  # Typical for modern GPUs
            'max_registers_per_sm': 131072,    # Typical for modern GPUs
        }
        
        print("✓ GPU occupancy information collected")
        return occupancy_info

class MockDataset(torch.utils.data.Dataset):
    """Mock dataset for occupancy testing"""
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
    """Simple model for occupancy testing"""
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

def test_occupancy_impact():
    """Test the impact of occupancy on performance"""
    print("Testing Occupancy Impact:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Different thread block sizes
    print("\n1. Thread Block Size Impact")
    sizes = [32, 64, 128, 256, 512, 1024]  # Different thread counts
    test_size = 1_000_000
    
    for thread_count in sizes:
        a = torch.randn(test_size, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Process in chunks to simulate thread blocks
        for i in range(0, test_size, thread_count):
            end = min(i + thread_count, test_size)
            a[i:end] *= 2.0
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"  {thread_count} threads: {elapsed*1000:.2f} ms")
    
    # Test 2: Register usage impact
    print("\n2. Register Usage Impact")
    size = 100_000
    
    # Low register usage
    torch.cuda.synchronize()
    start_time = time.time()
    a = torch.randn(size, device=device)
    b = a * 2.0
    torch.cuda.synchronize()
    low_reg_time = time.time() - start_time
    
    # High register usage (simulated with more operations)
    torch.cuda.synchronize()
    start_time = time.time()
    a = torch.randn(size, device=device)
    b = a * 2.0
    c = b * 3.0
    d = c * 4.0
    e = d * 5.0
    f = e * 6.0
    g = f * 7.0
    h = g * 8.0
    torch.cuda.synchronize()
    high_reg_time = time.time() - start_time
    
    print(f"  Low register usage: {low_reg_time*1000:.2f} ms")
    print(f"  High register usage: {high_reg_time*1000:.2f} ms")
    print(f"  Register overhead: {high_reg_time/low_reg_time:.1f}x")
    
    # Test 3: Shared memory impact
    print("\n3. Shared Memory Impact")
    size = 1_000_000
    
    # No shared memory usage
    torch.cuda.synchronize()
    start_time = time.time()
    a = torch.randn(size, device=device)
    b = a * 2.0
    torch.cuda.synchronize()
    no_shared_time = time.time() - start_time
    
    # Simulate shared memory usage with smaller chunks
    chunk_size = 1024  # Simulate shared memory
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(0, size, chunk_size):
        end = min(i + chunk_size, size)
        a[i:end] *= 2.0
    
    torch.cuda.synchronize()
    shared_time = time.time() - start_time
    
    print(f"  No shared memory: {no_shared_time*1000:.2f} ms")
    print(f"  With shared memory simulation: {shared_time*1000:.2f} ms")

def test_warp_efficiency():
    """Test warp efficiency and stall reasons"""
    print("\nTesting Warp Efficiency:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Memory-bound workload
    print("\n1. Memory-Bound Workload")
    size = 10_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c = a + b  # Memory-bound operation
    torch.cuda.synchronize()
    memory_time = time.time() - start_time
    
    bytes_moved = size * 4 * 3
    memory_bandwidth = bytes_moved / memory_time / (1024**3)
    
    print(f"  Memory-bound time: {memory_time*1000:.2f} ms")
    print(f"  Memory bandwidth: {memory_bandwidth:.1f} GB/s")
    
    # Test 2: Compute-bound workload
    print("\n2. Compute-Bound Workload")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Heavy computation
    for i in range(100):
        a = torch.sin(a) + torch.cos(a) + torch.tan(a)
    
    torch.cuda.synchronize()
    compute_time = time.time() - start_time
    
    print(f"  Compute-bound time: {compute_time*1000:.2f} ms")
    
    # Test 3: Synchronization overhead
    print("\n3. Synchronization Overhead")
    size = 100_000
    
    a = torch.randn(size, device=device)
    
    # Without synchronization
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(100):
        a *= 2.0
    
    torch.cuda.synchronize()
    no_sync_time = time.time() - start_time
    
    # With frequent synchronization
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(100):
        a *= 2.0
        torch.cuda.synchronize()  # Frequent sync
    
    sync_time = time.time() - start_time
    
    print(f"  No synchronization: {no_sync_time*1000:.2f} ms")
    print(f"  With synchronization: {sync_time*1000:.2f} ms")
    print(f"  Sync overhead: {sync_time/no_sync_time:.1f}x")

def test_instruction_level_parallelism():
    """Test instruction-level parallelism (ILP)"""
    print("\nTesting Instruction-Level Parallelism:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Sequential operations (low ILP)
    print("\n1. Sequential Operations (Low ILP)")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    c = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Sequential operations
    d = a + b
    e = d * c
    f = e / 2.0
    g = f ** 2
    
    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    
    print(f"  Sequential time: {sequential_time*1000:.2f} ms")
    
    # Test 2: Parallel operations (high ILP)
    print("\n2. Parallel Operations (High ILP)")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Parallel operations (independent)
    d = a + b
    e = c * 2.0
    f = a ** 2
    g = b / 3.0
    
    torch.cuda.synchronize()
    parallel_time = time.time() - start_time
    
    print(f"  Parallel time: {parallel_time*1000:.2f} ms")
    print(f"  ILP speedup: {sequential_time/parallel_time:.1f}x")
    
    # Test 3: Loop unrolling impact
    print("\n3. Loop Unrolling Impact")
    size = 100_000
    
    a = torch.randn(size, device=device)
    
    # Without unrolling
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(100):
        a = a * 2.0 + 1.0
    
    torch.cuda.synchronize()
    no_unroll_time = time.time() - start_time
    
    # With unrolling (simulated)
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Process in chunks to simulate unrolling
    chunk_size = 4
    for i in range(0, size, chunk_size):
        end = min(i + chunk_size, size)
        a[i:end] = a[i:end] * 2.0 + 1.0
    
    torch.cuda.synchronize()
    unroll_time = time.time() - start_time
    
    print(f"  No unrolling: {no_unroll_time*1000:.2f} ms")
    print(f"  With unrolling: {unroll_time*1000:.2f} ms")
    print(f"  Unrolling speedup: {no_unroll_time/unroll_time:.1f}x")

def test_triton_occupancy_optimization():
    """Test Triton-based occupancy optimization"""
    print("\nTesting Triton Occupancy Optimization:")
    print("=" * 50)
    
    if not TRITON_AVAILABLE:
        print("Triton not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Basic Triton kernel with occupancy optimization
    print("\n1. Basic Triton Kernel with Occupancy Optimization")
    
    @triton.jit
    def occupancy_optimized_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load with coalesced access
        x = tl.load(x_ptr + offsets, mask=mask)
        # Process with high ILP
        y = x * 2.0 + 1.0
        # Store with coalesced access
        tl.store(y_ptr + offsets, y, mask=mask)
    
    size = 1_000_000
    x = torch.randn(size, device=device)
    y = torch.empty_like(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    occupancy_optimized_kernel[grid](x, y, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  Triton occupancy optimized time: {elapsed*1000:.2f} ms")
    
    # Verify result
    expected = x * 2.0 + 1.0
    assert torch.allclose(y, expected, atol=1e-5)
    print("  ✓ Result verified")
    
    # Test 2: Different block sizes
    print("\n2. Different Block Sizes")
    block_sizes = [256, 512, 1024, 2048]
    
    for block_size in block_sizes:
        torch.cuda.synchronize()
        start_time = time.time()
        
        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        occupancy_optimized_kernel[grid](x, y, size, BLOCK_SIZE=block_size)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print(f"  Block size {block_size}: {elapsed*1000:.2f} ms")

def test_occupancy_profiling():
    """Test comprehensive occupancy profiling"""
    print("\nTesting Occupancy Profiling:")
    print("=" * 50)
    
    profiler = OccupancyProfiler()
    
    # Test 1: Basic occupancy profiling
    print("\n1. Basic Occupancy Profiling")
    basic_command = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from code.ch8.occupancy_warp_optimization import MockDataset, SimpleModel

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
    
    # Test 2: GPU occupancy analysis
    print("\n2. GPU Occupancy Analysis")
    occupancy_info = profiler.get_gpu_occupancy_info()
    if occupancy_info:
        print(f"  Device: {occupancy_info['device_name']}")
        print(f"  Compute Capability: {occupancy_info['compute_capability']}")
        print(f"  Max Threads per Block: {occupancy_info['max_threads_per_block']}")
        print(f"  Max Warps per SM: {occupancy_info['max_warps_per_sm']}")
        print(f"  Max Shared Memory per Block: {occupancy_info['max_shared_memory_per_block'] / 1024:.1f} KB")
        print(f"  Max Registers per Block: {occupancy_info['max_registers_per_block']}")
    
    # Test 3: PyTorch profiler with occupancy analysis
    print("\n3. PyTorch Profiler with Occupancy Analysis")
    dataset = MockDataset(100)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    model = SimpleModel().cuda()
    
    pytorch_prof = profiler.run_pytorch_profiler(model, dataloader, num_batches=5)
    
    # Test 4: Memory profiling
    print("\n4. Occupancy Memory Profiling")
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
        'occupancy_info': occupancy_info,
        'pytorch_prof': pytorch_prof,
        'memory_prof': memory_prof
    }

def test_roofline_analysis():
    """Test roofline analysis for performance bottlenecks"""
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
    flops = size  # 1 FLOP per element
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
        a = torch.sin(a) + torch.cos(a) + torch.tan(a)
    
    torch.cuda.synchronize()
    compute_time = time.time() - start_time
    
    flops = size * 100 * 3  # 3 FLOPs per element per iteration
    bytes_moved = size * 4 * 2  # Rough estimate
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
        a = c * 2.0
    
    torch.cuda.synchronize()
    balanced_time = time.time() - start_time
    
    flops = size * 10 * 4  # 4 FLOPs per element per iteration
    bytes_moved = size * 4 * 3 * 10  # Rough estimate
    arithmetic_intensity = flops / bytes_moved
    
    print(f"  Balanced time: {balanced_time*1000:.2f} ms")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")

def main():
    """Main function demonstrating occupancy and warp optimization"""
    parser = argparse.ArgumentParser(description="Occupancy and Warp Optimization Examples")
    parser.add_argument("--test", choices=["all", "occupancy", "warp", "ilp", "triton", "profiling", "roofline"], 
                       default="all", help="Test to run")
    args = parser.parse_args()
    
    print("AI Performance Engineering - Chapter 8")
    print("Occupancy and Warp Optimization with Comprehensive Profiling")
    print("=" * 70)
    
    if args.test == "occupancy" or args.test == "all":
        test_occupancy_impact()
    
    if args.test == "warp" or args.test == "all":
        test_warp_efficiency()
    
    if args.test == "ilp" or args.test == "all":
        test_instruction_level_parallelism()
    
    if args.test == "triton" or args.test == "all":
        test_triton_occupancy_optimization()
    
    if args.test == "profiling" or args.test == "all":
        test_occupancy_profiling()
    
    if args.test == "roofline" or args.test == "all":
        test_roofline_analysis()
    
    print("\nOccupancy and Warp Optimization Summary:")
    print("=" * 50)
    print("✓ Optimize occupancy for better latency hiding")
    print("✓ Balance register and shared memory usage")
    print("✓ Increase instruction-level parallelism (ILP)")
    print("✓ Profile with nsys, ncu, PyTorch profiler, and memory profiler")
    print("✓ Use Triton for custom occupancy-optimized kernels")
    print("✓ Understand warp stall reasons and roofline analysis")
    print("✓ Monitor occupancy, memory bandwidth, and compute utilization")

if __name__ == "__main__":
    main()
