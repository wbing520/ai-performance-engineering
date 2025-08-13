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

class MemoryAccessProfiler:
    """Comprehensive memory access profiling with multiple tools"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiling_results = {}
        
    def run_nsys_profile(self, command, output_file="memory_access_profile"):
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
    
    def run_ncu_profile(self, command, output_file="memory_access_ncu"):
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
        """Run PyTorch profiler with detailed memory analysis"""
        print("Running PyTorch profiler with memory analysis...")
        
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
            on_trace_ready=profiler.tensorboard_trace_handler('./log/memory_profiler'),
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
        """Run PyTorch memory profiler for detailed memory analysis"""
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
from code.ch7.memory_access_optimization import SimpleModel, MockDataset

# Setup
dataset = MockDataset(100)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
model = SimpleModel().cuda()

# HTA-style profiling
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler('./log/hta_memory_profile'),
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
            
print("HTA memory profiling completed")
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
        """Run Linux perf profiling for system-level memory analysis"""
        print(f"Running perf profiling: {command}")
        
        perf_cmd = [
            "perf", "record", 
            "-g",  # Call graph
            "-F", "99",  # 99 Hz sampling
            "-o", "perf_memory.data",
            "python", "-c", command
        ]
        
        try:
            result = subprocess.run(perf_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Generate report
                report_cmd = ["perf", "report", "-i", "perf_memory.data"]
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
    """Mock dataset for memory access testing"""
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
    """Simple model for memory access testing"""
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

def test_coalesced_vs_uncoalesced_access():
    """Test coalesced vs uncoalesced memory access patterns"""
    print("Testing Coalesced vs Uncoalesced Memory Access:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Coalesced access (contiguous)
    print("\n1. Coalesced Access (Contiguous)")
    size = 10_000_000
    
    # Coalesced: sequential access
    a = torch.randn(size, device=device)
    torch.cuda.synchronize()
    start_time = time.time()
    b = a * 2.0  # Contiguous access
    torch.cuda.synchronize()
    coalesced_time = time.time() - start_time
    
    bytes_moved = size * 4 * 2  # 2 tensors, 4 bytes each
    coalesced_bandwidth = bytes_moved / coalesced_time / (1024**3)
    
    print(f"  Coalesced time: {coalesced_time*1000:.2f} ms")
    print(f"  Coalesced bandwidth: {coalesced_bandwidth:.1f} GB/s")
    
    # Test 2: Uncoalesced access (strided)
    print("\n2. Uncoalesced Access (Strided)")
    stride = 2
    indices = torch.arange(0, size, stride, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c = a[indices] * 2.0  # Strided access
    torch.cuda.synchronize()
    uncoalesced_time = time.time() - start_time
    
    bytes_moved = len(indices) * 4 * 2
    uncoalesced_bandwidth = bytes_moved / uncoalesced_time / (1024**3)
    
    print(f"  Uncoalesced time: {uncoalesced_time*1000:.2f} ms")
    print(f"  Uncoalesced bandwidth: {uncoalesced_bandwidth:.1f} GB/s")
    print(f"  Speedup: {uncoalesced_time/coalesced_time:.1f}x")
    
    # Test 3: Random access (worst case)
    print("\n3. Random Access (Worst Case)")
    random_indices = torch.randperm(size, device=device)[:size//2]
    
    torch.cuda.synchronize()
    start_time = time.time()
    d = a[random_indices] * 2.0  # Random access
    torch.cuda.synchronize()
    random_time = time.time() - start_time
    
    bytes_moved = len(random_indices) * 4 * 2
    random_bandwidth = bytes_moved / random_time / (1024**3)
    
    print(f"  Random time: {random_time*1000:.2f} ms")
    print(f"  Random bandwidth: {random_bandwidth:.1f} GB/s")
    print(f"  Speedup vs coalesced: {random_time/coalesced_time:.1f}x")

def test_vectorized_memory_access():
    """Test vectorized memory access patterns"""
    print("\nTesting Vectorized Memory Access:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Scalar access (float32)
    print("\n1. Scalar Access (float32)")
    size = 10_000_000
    
    a = torch.randn(size, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    start_time = time.time()
    b = a * 2.0
    torch.cuda.synchronize()
    scalar_time = time.time() - start_time
    
    bytes_moved = size * 4 * 2
    scalar_bandwidth = bytes_moved / scalar_time / (1024**3)
    
    print(f"  Scalar time: {scalar_time*1000:.2f} ms")
    print(f"  Scalar bandwidth: {scalar_bandwidth:.1f} GB/s")
    
    # Test 2: Vectorized access (float16)
    print("\n2. Vectorized Access (float16)")
    a_half = torch.randn(size, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    start_time = time.time()
    b_half = a_half * 2.0
    torch.cuda.synchronize()
    vector_time = time.time() - start_time
    
    bytes_moved = size * 2 * 2  # float16 = 2 bytes
    vector_bandwidth = bytes_moved / vector_time / (1024**3)
    
    print(f"  Vector time: {vector_time*1000:.2f} ms")
    print(f"  Vector bandwidth: {vector_bandwidth:.1f} GB/s")
    print(f"  Speedup: {scalar_time/vector_time:.1f}x")
    
    # Test 3: Structured data access
    print("\n3. Structured Data Access")
    # Array of Structs (AoS) vs Structure of Arrays (SoA)
    
    # AoS: [[x1,y1,z1], [x2,y2,z2], ...]
    aos_size = size // 3
    aos_data = torch.randn(aos_size, 3, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    aos_result = aos_data * 2.0
    torch.cuda.synchronize()
    aos_time = time.time() - start_time
    
    # SoA: [x1,x2,...], [y1,y2,...], [z1,z2,...]
    soa_x = torch.randn(aos_size, device=device)
    soa_y = torch.randn(aos_size, device=device)
    soa_z = torch.randn(aos_size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    soa_result_x = soa_x * 2.0
    soa_result_y = soa_y * 2.0
    soa_result_z = soa_z * 2.0
    torch.cuda.synchronize()
    soa_time = time.time() - start_time
    
    print(f"  AoS time: {aos_time*1000:.2f} ms")
    print(f"  SoA time: {soa_time*1000:.2f} ms")
    print(f"  SoA speedup: {aos_time/soa_time:.1f}x")

def test_memory_hierarchy_optimization():
    """Test memory hierarchy optimization techniques"""
    print("\nTesting Memory Hierarchy Optimization:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Shared memory simulation
    print("\n1. Shared Memory Simulation")
    size = 1_000_000
    
    # Global memory access
    a = torch.randn(size, device=device)
    torch.cuda.synchronize()
    start_time = time.time()
    b = a * 2.0
    torch.cuda.synchronize()
    global_time = time.time() - start_time
    
    # Simulate shared memory with smaller chunks
    chunk_size = 1024  # Simulate shared memory size
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(0, size, chunk_size):
        end = min(i + chunk_size, size)
        a[i:end] *= 2.0
    
    torch.cuda.synchronize()
    shared_time = time.time() - start_time
    
    print(f"  Global memory time: {global_time*1000:.2f} ms")
    print(f"  Shared memory simulation time: {shared_time*1000:.2f} ms")
    
    # Test 2: Memory alignment
    print("\n2. Memory Alignment Test")
    size = 10_000_000
    
    # Aligned access
    a_aligned = torch.randn(size, device=device)
    torch.cuda.synchronize()
    start_time = time.time()
    b_aligned = a_aligned * 2.0
    torch.cuda.synchronize()
    aligned_time = time.time() - start_time
    
    # Unaligned access (offset by 1)
    a_unaligned = torch.randn(size + 1, device=device)[1:]  # Offset by 1
    torch.cuda.synchronize()
    start_time = time.time()
    b_unaligned = a_unaligned * 2.0
    torch.cuda.synchronize()
    unaligned_time = time.time() - start_time
    
    print(f"  Aligned time: {aligned_time*1000:.2f} ms")
    print(f"  Unaligned time: {unaligned_time*1000:.2f} ms")
    print(f"  Alignment overhead: {unaligned_time/aligned_time:.1f}x")
    
    # Test 3: Memory bandwidth by data type
    print("\n3. Memory Bandwidth by Data Type")
    size = 10_000_000
    
    for dtype in [torch.float32, torch.float16, torch.int32, torch.int16]:
        a = torch.randn(size, dtype=dtype, device=device)
        b = torch.randn(size, dtype=dtype, device=device)
        
        torch.cuda.synchronize()
        start_time = time.time()
        c = a + b
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        bytes_moved = size * a.element_size() * 3
        bandwidth_gb_s = bytes_moved / elapsed / (1024**3)
        print(f"  {dtype}: {bandwidth_gb_s:.1f} GB/s")

def test_triton_memory_optimization():
    """Test Triton-based memory optimization"""
    print("\nTesting Triton Memory Optimization:")
    print("=" * 50)
    
    if not TRITON_AVAILABLE:
        print("Triton not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Basic Triton kernel with memory optimization
    print("\n1. Basic Triton Kernel with Memory Optimization")
    
    @triton.jit
    def optimized_copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load with coalesced access
        x = tl.load(x_ptr + offsets, mask=mask)
        # Store with coalesced access
        tl.store(y_ptr + offsets, x, mask=mask)
    
    size = 1_000_000
    x = torch.randn(size, device=device)
    y = torch.empty_like(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    optimized_copy_kernel[grid](x, y, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  Triton optimized copy time: {elapsed*1000:.2f} ms")
    
    # Verify result
    assert torch.allclose(x, y, atol=1e-5)
    print("  ✓ Result verified")
    
    # Test 2: Vectorized Triton kernel
    print("\n2. Vectorized Triton Kernel")
    
    @triton.jit
    def vectorized_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load 4 elements at once (float4 equivalent)
        x = tl.load(x_ptr + offsets, mask=mask)
        # Process
        y = x * 2.0
        # Store 4 elements at once
        tl.store(y_ptr + offsets, y, mask=mask)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    vectorized_kernel[grid](x, y, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  Vectorized Triton kernel time: {elapsed*1000:.2f} ms")

def test_memory_profiling():
    """Test comprehensive memory profiling"""
    print("\nTesting Memory Profiling:")
    print("=" * 50)
    
    profiler = MemoryAccessProfiler()
    
    # Test 1: Basic memory profiling
    print("\n1. Basic Memory Profiling")
    basic_command = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from code.ch7.memory_access_optimization import MockDataset, SimpleModel

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
    
    # Test 2: PyTorch profiler with memory analysis
    print("\n2. PyTorch Profiler with Memory Analysis")
    dataset = MockDataset(100)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    model = SimpleModel().cuda()
    
    pytorch_prof = profiler.run_pytorch_profiler(model, dataloader, num_batches=5)
    
    # Test 3: Memory profiling
    print("\n3. Memory Profiling")
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

def test_memory_access_patterns():
    """Test different memory access patterns"""
    print("\nTesting Memory Access Patterns:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Sequential access
    print("\n1. Sequential Access")
    size = 10_000_000
    a = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    b = a * 2.0
    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    
    print(f"  Sequential time: {sequential_time*1000:.2f} ms")
    
    # Test 2: Strided access
    print("\n2. Strided Access")
    stride = 2
    indices = torch.arange(0, size, stride, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    c = a[indices] * 2.0
    torch.cuda.synchronize()
    strided_time = time.time() - start_time
    
    print(f"  Strided time: {strided_time*1000:.2f} ms")
    print(f"  Strided overhead: {strided_time/sequential_time:.1f}x")
    
    # Test 3: Random access
    print("\n3. Random Access")
    random_indices = torch.randperm(size, device=device)[:size//2]
    
    torch.cuda.synchronize()
    start_time = time.time()
    d = a[random_indices] * 2.0
    torch.cuda.synchronize()
    random_time = time.time() - start_time
    
    print(f"  Random time: {random_time*1000:.2f} ms")
    print(f"  Random overhead: {random_time/sequential_time:.1f}x")
    
    # Test 4: Block access
    print("\n4. Block Access")
    block_size = 1024
    torch.cuda.synchronize()
    start_time = time.time()
    
    for i in range(0, size, block_size):
        end = min(i + block_size, size)
        a[i:end] *= 2.0
    
    torch.cuda.synchronize()
    block_time = time.time() - start_time
    
    print(f"  Block time: {block_time*1000:.2f} ms")
    print(f"  Block overhead: {block_time/sequential_time:.1f}x")

def main():
    """Main function demonstrating memory access optimization"""
    parser = argparse.ArgumentParser(description="Memory Access Optimization Examples")
    parser.add_argument("--test", choices=["all", "coalesced", "vectorized", "hierarchy", "triton", "profiling", "patterns"], 
                       default="all", help="Test to run")
    args = parser.parse_args()
    
    print("AI Performance Engineering - Chapter 7")
    print("Memory Access Optimization with Comprehensive Profiling")
    print("=" * 70)
    
    if args.test == "coalesced" or args.test == "all":
        test_coalesced_vs_uncoalesced_access()
    
    if args.test == "vectorized" or args.test == "all":
        test_vectorized_memory_access()
    
    if args.test == "hierarchy" or args.test == "all":
        test_memory_hierarchy_optimization()
    
    if args.test == "triton" or args.test == "all":
        test_triton_memory_optimization()
    
    if args.test == "profiling" or args.test == "all":
        test_memory_profiling()
    
    if args.test == "patterns" or args.test == "all":
        test_memory_access_patterns()
    
    print("\nMemory Access Optimization Summary:")
    print("=" * 50)
    print("✓ Use coalesced memory access for optimal bandwidth")
    print("✓ Implement vectorized loads (float4, etc.) for efficiency")
    print("✓ Choose SoA over AoS for better cache utilization")
    print("✓ Align memory accesses to cache line boundaries")
    print("✓ Profile with nsys, ncu, PyTorch profiler, and memory profiler")
    print("✓ Use Triton for custom memory-optimized kernels")
    print("✓ Understand memory hierarchy (L1, L2, HBM)")
    print("✓ Monitor memory bandwidth and access patterns")

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
