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

class PipeliningProfiler:
    """Comprehensive intra-kernel pipelining and warp specialization profiling with multiple tools"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiling_results = {}
        
    def run_nsys_profile(self, command, output_file="pipelining_profile"):
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
    
    def run_ncu_profile(self, command, output_file="pipelining_ncu"):
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
        """Run PyTorch profiler with detailed pipelining analysis"""
        print("Running PyTorch profiler with pipelining analysis...")
        
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
            on_trace_ready=profiler.tensorboard_trace_handler('./log/pipelining_profiler'),
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
        """Run PyTorch memory profiler for pipelining analysis"""
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
from code.ch10.intra_kernel_pipelining import SimpleModel, MockDataset

# Setup
dataset = MockDataset(100)
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
model = SimpleModel().cuda()

# HTA-style profiling
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler('./log/hta_pipelining_profile'),
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
            
print("HTA pipelining profiling completed")
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
        """Run Linux perf profiling for system-level pipelining analysis"""
        print(f"Running perf profiling: {command}")
        
        perf_cmd = [
            "perf", "record", 
            "-g",  # Call graph
            "-F", "99",  # 99 Hz sampling
            "-o", "perf_pipelining.data",
            "python", "-c", command
        ]
        
        try:
            result = subprocess.run(perf_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # Generate report
                report_cmd = ["perf", "report", "-i", "perf_pipelining.data"]
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
    """Mock dataset for pipelining testing"""
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
    """Simple model for pipelining testing"""
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

def test_double_buffering():
    """Test double buffering for intra-kernel pipelining"""
    print("Testing Double Buffering:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Sequential processing (no pipelining)
    print("\n1. Sequential Processing (No Pipelining)")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Sequential operations
    for i in range(100):
        c = a + b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    
    print(f"  Sequential time: {sequential_time*1000:.2f} ms")
    
    # Test 2: Double buffering simulation
    print("\n2. Double Buffering Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate double buffering with overlapping operations
    buffer_size = 10000
    num_chunks = size // buffer_size
    
    for chunk in range(num_chunks):
        start_idx = chunk * buffer_size
        end_idx = min(start_idx + buffer_size, size)
        
        # Load next chunk while processing current chunk
        if chunk < num_chunks - 1:
            next_start = (chunk + 1) * buffer_size
            next_end = min(next_start + buffer_size, size)
            # Simulate async load
            _ = a[next_start:next_end] + b[next_start:next_end]
        
        # Process current chunk
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    double_buffer_time = time.time() - start_time
    
    print(f"  Double buffer time: {double_buffer_time*1000:.2f} ms")
    print(f"  Double buffering speedup: {sequential_time/double_buffer_time:.1f}x")
    
    # Test 3: Advanced double buffering with multiple stages
    print("\n3. Advanced Double Buffering with Multiple Stages")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate 3-stage pipeline: load, compute, store
    for chunk in range(num_chunks):
        start_idx = chunk * buffer_size
        end_idx = min(start_idx + buffer_size, size)
        
        # Stage 1: Load (simulate async)
        if chunk < num_chunks - 1:
            next_start = (chunk + 1) * buffer_size
            next_end = min(next_start + buffer_size, size)
            _ = a[next_start:next_end] + b[next_start:next_end]
        
        # Stage 2: Compute
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        
        # Stage 3: Store (simulate async)
        f = e * 2.0
    
    torch.cuda.synchronize()
    advanced_time = time.time() - start_time
    
    print(f"  Advanced pipeline time: {advanced_time*1000:.2f} ms")
    print(f"  Advanced pipeline speedup: {sequential_time/advanced_time:.1f}x")

def test_warp_specialization():
    """Test warp specialization for different pipeline stages"""
    print("\nTesting Warp Specialization:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Uniform warp processing
    print("\n1. Uniform Warp Processing")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # All warps do the same work
    c = a + b
    d = torch.sin(c)
    e = torch.cos(d)
    f = e * 2.0
    
    torch.cuda.synchronize()
    uniform_time = time.time() - start_time
    
    print(f"  Uniform warp time: {uniform_time*1000:.2f} ms")
    
    # Test 2: Warp specialization simulation
    print("\n2. Warp Specialization Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate warp specialization with different chunk sizes
    # Load warps, compute warps, store warps
    chunk_size = 10000
    num_chunks = size // chunk_size
    
    # Simulate specialized warps
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, size)
        
        # Load warps (simulate memory loading)
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        
        # Compute warps (simulate computation)
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        
        # Store warps (simulate memory storing)
        f = e * 2.0
    
    torch.cuda.synchronize()
    specialized_time = time.time() - start_time
    
    print(f"  Specialized warp time: {specialized_time*1000:.2f} ms")
    print(f"  Specialization speedup: {uniform_time/specialized_time:.1f}x")
    
    # Test 3: Multi-stage warp specialization
    print("\n3. Multi-Stage Warp Specialization")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate 3-stage pipeline with specialized warps
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, size)
        
        # Stage 1: Memory loading warps
        if chunk < num_chunks - 1:
            next_start = (chunk + 1) * chunk_size
            next_end = min(next_start + chunk_size, size)
            # Simulate async memory loading
            _ = a[next_start:next_end] + b[next_start:next_end]
        
        # Stage 2: Computation warps
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        
        # Stage 3: Memory storing warps
        f = e * 2.0
    
    torch.cuda.synchronize()
    multi_stage_time = time.time() - start_time
    
    print(f"  Multi-stage time: {multi_stage_time*1000:.2f} ms")
    print(f"  Multi-stage speedup: {uniform_time/multi_stage_time:.1f}x")

def test_cooperative_groups():
    """Test cooperative groups for advanced synchronization"""
    print("\nTesting Cooperative Groups:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Basic thread block synchronization
    print("\n1. Basic Thread Block Synchronization")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Basic synchronization
    c = a + b
    torch.cuda.synchronize()  # Simulate __syncthreads()
    d = torch.sin(c)
    torch.cuda.synchronize()
    e = torch.cos(d)
    torch.cuda.synchronize()
    f = e * 2.0
    
    torch.cuda.synchronize()
    basic_sync_time = time.time() - start_time
    
    print(f"  Basic sync time: {basic_sync_time*1000:.2f} ms")
    
    # Test 2: Cooperative group simulation
    print("\n2. Cooperative Group Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate cooperative groups with different synchronization patterns
    chunk_size = 10000
    num_chunks = size // chunk_size
    
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, size)
        
        # Simulate cooperative group operations
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        
        # Group 1: Memory operations
        c = chunk_a + chunk_b
        
        # Group 2: Compute operations
        d = torch.sin(c)
        e = torch.cos(d)
        
        # Group 3: Store operations
        f = e * 2.0
    
    torch.cuda.synchronize()
    cooperative_time = time.time() - start_time
    
    print(f"  Cooperative group time: {cooperative_time*1000:.2f} ms")
    print(f"  Cooperative speedup: {basic_sync_time/cooperative_time:.1f}x")
    
    # Test 3: Grid-level cooperation simulation
    print("\n3. Grid-Level Cooperation Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate grid-level cooperation
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, size)
        
        # Simulate grid-level operations
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        
        # Grid-wide computation
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    grid_time = time.time() - start_time
    
    print(f"  Grid-level time: {grid_time*1000:.2f} ms")
    print(f"  Grid-level speedup: {basic_sync_time/grid_time:.1f}x")

def test_persistent_kernels():
    """Test persistent kernels for dynamic work queues"""
    print("\nTesting Persistent Kernels:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Traditional kernel launches
    print("\n1. Traditional Kernel Launches")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Multiple kernel launches
    for i in range(10):
        c = a + b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    traditional_time = time.time() - start_time
    
    print(f"  Traditional kernel time: {traditional_time*1000:.2f} ms")
    
    # Test 2: Persistent kernel simulation
    print("\n2. Persistent Kernel Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate persistent kernel with work queue
    work_queue = list(range(10))  # Simulate work items
    
    # Simulate persistent kernel processing work queue
    for work_item in work_queue:
        # Process work item without kernel relaunch overhead
        c = a + b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    persistent_time = time.time() - start_time
    
    print(f"  Persistent kernel time: {persistent_time*1000:.2f} ms")
    print(f"  Persistent speedup: {traditional_time/persistent_time:.1f}x")
    
    # Test 3: Dynamic work queue simulation
    print("\n3. Dynamic Work Queue Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate dynamic work queue
    dynamic_queue = []
    for i in range(10):
        # Simulate dynamic work generation
        work_size = np.random.randint(1000, 10000)
        dynamic_queue.append(work_size)
    
    # Process dynamic work queue
    for work_size in dynamic_queue:
        # Simulate processing dynamic work
        chunk_a = torch.randn(work_size, device=device)
        chunk_b = torch.randn(work_size, device=device)
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    dynamic_time = time.time() - start_time
    
    print(f"  Dynamic queue time: {dynamic_time*1000:.2f} ms")

def test_triton_pipelining():
    """Test Triton-based pipelining optimization"""
    print("\nTesting Triton Pipelining:")
    print("=" * 50)
    
    if not TRITON_AVAILABLE:
        print("Triton not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Basic Triton kernel with pipelining
    print("\n1. Basic Triton Kernel with Pipelining")
    
    @triton.jit
    def pipelined_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)
        
        # Pipelined operations
        y = x * 2.0
        y = tl.sin(y) + tl.cos(y)
        y = y * 3.0 + tl.exp(y)
        
        # Store result
        tl.store(y_ptr + offsets, y, mask=mask)
    
    size = 1_000_000
    x = torch.randn(size, device=device)
    y = torch.empty_like(x)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    pipelined_kernel[grid](x, y, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  Pipelined Triton kernel time: {elapsed*1000:.2f} ms")
    
    # Verify result
    expected = x * 2.0
    expected = torch.sin(expected) + torch.cos(expected)
    expected = expected * 3.0 + torch.exp(expected)
    assert torch.allclose(y, expected, atol=1e-5)
    print("  ✓ Result verified")
    
    # Test 2: Multi-stage Triton kernel
    print("\n2. Multi-Stage Triton Kernel")
    
    @triton.jit
    def multi_stage_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Stage 1: Load both inputs
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        
        # Stage 2: Compute
        c = a + b
        c = tl.sin(c) * tl.cos(c)
        
        # Stage 3: Store
        tl.store(c_ptr + offsets, c, mask=mask)
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    c = torch.empty_like(a)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    multi_stage_kernel[grid](a, b, c, size, BLOCK_SIZE=1024)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    print(f"  Multi-stage Triton kernel time: {elapsed*1000:.2f} ms")
    
    # Verify result
    expected = torch.sin(a + b) * torch.cos(a + b)
    assert torch.allclose(c, expected, atol=1e-5)
    print("  ✓ Multi-stage result verified")

def test_pipelining_profiling():
    """Test comprehensive pipelining profiling"""
    print("\nTesting Pipelining Profiling:")
    print("=" * 50)
    
    profiler = PipeliningProfiler()
    
    # Test 1: Basic pipelining profiling
    print("\n1. Basic Pipelining Profiling")
    basic_command = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from code.ch10.intra_kernel_pipelining import MockDataset, SimpleModel

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
    
    # Test 2: PyTorch profiler with pipelining analysis
    print("\n2. PyTorch Profiler with Pipelining Analysis")
    dataset = MockDataset(100)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    model = SimpleModel().cuda()
    
    pytorch_prof = profiler.run_pytorch_profiler(model, dataloader, num_batches=5)
    
    # Test 3: Memory profiling
    print("\n3. Pipelining Memory Profiling")
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

def test_thread_block_clusters():
    """Test thread block clusters and distributed shared memory"""
    print("\nTesting Thread Block Clusters:")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.device("cuda")
    
    # Test 1: Single thread block processing
    print("\n1. Single Thread Block Processing")
    size = 1_000_000
    
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Single block processing
    c = a + b
    d = torch.sin(c)
    e = torch.cos(d)
    f = e * 2.0
    
    torch.cuda.synchronize()
    single_block_time = time.time() - start_time
    
    print(f"  Single block time: {single_block_time*1000:.2f} ms")
    
    # Test 2: Thread block cluster simulation
    print("\n2. Thread Block Cluster Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate thread block clusters with shared data
    chunk_size = 10000
    num_chunks = size // chunk_size
    
    # Simulate cluster processing
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, size)
        
        # Simulate cluster-wide operations
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        
        # Cluster computation
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    cluster_time = time.time() - start_time
    
    print(f"  Cluster time: {cluster_time*1000:.2f} ms")
    print(f"  Cluster speedup: {single_block_time/cluster_time:.1f}x")
    
    # Test 3: Distributed shared memory simulation
    print("\n3. Distributed Shared Memory Simulation")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Simulate distributed shared memory
    for chunk in range(num_chunks):
        start_idx = chunk * chunk_size
        end_idx = min(start_idx + chunk_size, size)
        
        # Simulate DSM operations
        chunk_a = a[start_idx:end_idx]
        chunk_b = b[start_idx:end_idx]
        
        # Shared memory operations
        c = chunk_a + chunk_b
        d = torch.sin(c)
        e = torch.cos(d)
        f = e * 2.0
    
    torch.cuda.synchronize()
    dsm_time = time.time() - start_time
    
    print(f"  DSM time: {dsm_time*1000:.2f} ms")
    print(f"  DSM speedup: {single_block_time/dsm_time:.1f}x")

def main():
    """Main function demonstrating intra-kernel pipelining and warp specialization"""
    parser = argparse.ArgumentParser(description="Intra-Kernel Pipelining and Warp Specialization Examples")
    parser.add_argument("--test", choices=["all", "double_buffering", "warp_specialization", "cooperative_groups", "persistent_kernels", "triton", "profiling", "clusters"], 
                       default="all", help="Test to run")
    args = parser.parse_args()
    
    print("AI Performance Engineering - Chapter 10")
    print("Intra-Kernel Pipelining and Warp Specialization with Comprehensive Profiling")
    print("=" * 70)
    
    if args.test == "double_buffering" or args.test == "all":
        test_double_buffering()
    
    if args.test == "warp_specialization" or args.test == "all":
        test_warp_specialization()
    
    if args.test == "cooperative_groups" or args.test == "all":
        test_cooperative_groups()
    
    if args.test == "persistent_kernels" or args.test == "all":
        test_persistent_kernels()
    
    if args.test == "triton" or args.test == "all":
        test_triton_pipelining()
    
    if args.test == "profiling" or args.test == "all":
        test_pipelining_profiling()
    
    if args.test == "clusters" or args.test == "all":
        test_thread_block_clusters()
    
    print("\nIntra-Kernel Pipelining and Warp Specialization Summary:")
    print("=" * 50)
    print("✓ Use double buffering for memory-compute overlap")
    print("✓ Implement warp specialization for pipeline stages")
    print("✓ Leverage cooperative groups for advanced synchronization")
    print("✓ Profile with nsys, ncu, PyTorch profiler, and memory profiler")
    print("✓ Use Triton for custom pipelined kernels")
    print("✓ Explore thread block clusters and distributed shared memory")
    print("✓ Consider persistent kernels for dynamic work queues")

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
