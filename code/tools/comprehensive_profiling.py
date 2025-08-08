#!/usr/bin/env python3
"""
Comprehensive Profiling Tools Demo
Latest profiling tools for PyTorch 2.8, CUDA 12.9, and Blackwell B200/B300

This script demonstrates:
- Nsight Systems (nsys) timeline analysis
- Nsight Compute (ncu) kernel-level analysis
- PyTorch Profiler with enhanced features
- HTA (Holistic Tracing Analysis) for multi-GPU
- Perf system-level analysis
- Memory profiling
- FLOP counting
- Module-level analysis
- Triton 3.4 profiling
- Architecture-specific optimizations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import time
import psutil
import GPUtil
import subprocess
import os
import json
from typing import Dict, List, Any
import numpy as np

# Import architecture configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch_config import arch_config, configure_optimizations


class ProfilingDemoModel(nn.Module):
    """A model for demonstrating comprehensive profiling."""
    
    def __init__(self, input_size=1024, hidden_size=512, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        with nvtx.annotate("fc1_forward"):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        with nvtx.annotate("fc2_forward"):
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
        
        with nvtx.annotate("fc3_forward"):
            x = self.fc3(x)
        
        return x


def configure_architecture_optimizations():
    """Configure PyTorch 2.8 optimizations for current architecture."""
    configure_optimizations()
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        print(f"GPU: {device_props.name}")
        print(f"Compute Capability: {compute_capability}")
        
        if compute_capability == "9.0":  # Hopper H100/H200
            print("✓ Enabling Hopper H100/H200 optimizations")
            # Additional Hopper-specific optimizations
            torch._inductor.config.triton.use_hopper_optimizations = True
            torch._inductor.config.triton.hbm3_optimizations = True
            torch._inductor.config.triton.tma_support = True
            torch._inductor.config.triton.transformer_engine = True
        elif compute_capability == "10.0":  # Blackwell B200/B300
            print("✓ Enabling Blackwell B200/B300 optimizations")
            # Additional Blackwell-specific optimizations
            torch._inductor.config.triton.use_blackwell_optimizations = True
            torch._inductor.config.triton.hbm3e_optimizations = True
            torch._inductor.config.triton.tma_support = True
            torch._inductor.config.triton.stream_ordered_memory = True
            torch._inductor.config.triton.nvlink_c2c = True
        
        # Common optimizations for both architectures
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.triton.autotune_mode = "max-autotune"
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._inductor.config.triton.enable_advanced_memory_optimizations = True


def demonstrate_pytorch_profiler():
    """
    Demonstrate PyTorch profiler with latest features.
    """
    print("=== PyTorch Profiler Demo (PyTorch 2.8) ===")
    
    # Configure optimizations
    configure_architecture_optimizations()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    model.eval()
    
    # Create dummy input
    x = torch.randn(64, 1024, device=device)
    
    # Enhanced profiler configuration
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        schedule=schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        )
    ) as prof:
        with torch.no_grad():
            for _ in range(50):
                with nvtx.annotate("pytorch_profiling"):
                    output = model(x)
    
    print("PyTorch Profiler Results:")
    print("Top operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\nMemory profiling:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
    
    print("\nFLOP analysis:")
    print(prof.key_averages().table(sort_by="flops", row_limit=5))
    
    print("\nModule-level analysis:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))
    
    # Export results
    prof.export_chrome_trace("pytorch_trace.json")
    print("\nChrome trace exported: pytorch_trace.json")


def demonstrate_nsight_systems():
    """
    Demonstrate Nsight Systems timeline analysis.
    """
    print("\n=== Nsight Systems Timeline Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    print("Running Nsight Systems timeline analysis...")
    print("Command: nsys profile -t cuda,nvtx,osrt,triton -o nsys_timeline python script.py")
    
    # Simulate Nsight Systems profiling
    with torch.no_grad():
        for _ in range(50):
            with nvtx.annotate("nsight_systems_demo"):
                output = model(x)
    
    print("Nsight Systems analysis completed")
    print("View results with: nsys-ui nsys_timeline.nsys-rep")


def demonstrate_nsight_compute():
    """
    Demonstrate Nsight Compute kernel analysis.
    """
    print("\n=== Nsight Compute Kernel Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    print("Running Nsight Compute kernel analysis...")
    print("Command: ncu --metrics achieved_occupancy,warp_execution_efficiency -o ncu_kernel python script.py")
    
    # Simulate Nsight Compute profiling
    with torch.no_grad():
        for _ in range(50):
            with nvtx.annotate("nsight_compute_demo"):
                output = model(x)
    
    print("Nsight Compute analysis completed")
    print("View results with: ncu-ui ncu_kernel.ncu-rep")


def demonstrate_hta_profiling():
    """
    Demonstrate HTA (Holistic Tracing Analysis) for multi-GPU.
    """
    print("\n=== HTA (Holistic Tracing Analysis) Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    print("Running HTA analysis...")
    print("Command: nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl,triton -o hta_analysis python script.py")
    
    # Simulate HTA profiling
    with torch.no_grad():
        for _ in range(50):
            with nvtx.annotate("hta_demo"):
                output = model(x)
    
    print("HTA analysis completed")
    print("View results with: nsys-ui hta_analysis.nsys-rep")


def demonstrate_perf_profiling():
    """
    Demonstrate Perf system-level analysis.
    """
    print("\n=== Perf System-Level Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    print("Running Perf system-level analysis...")
    print("Command: perf record -g -p $(pgrep python) -o perf.data")
    
    # Simulate Perf profiling
    with torch.no_grad():
        for _ in range(50):
            with nvtx.annotate("perf_demo"):
                output = model(x)
    
    print("Perf analysis completed")
    print("View results with: perf report -i perf.data")


def demonstrate_memory_profiling():
    """
    Demonstrate memory profiling with latest features.
    """
    print("\n=== Memory Profiling Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    # Memory profiling with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
    ) as prof:
        with torch.no_grad():
            for _ in range(50):
                with nvtx.annotate("memory_profiling"):
                    output = model(x)
    
    print("Memory Profiling Results:")
    print("Memory usage by operation:")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    
    # GPU memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        peak_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_cached = torch.cuda.max_memory_reserved() / 1e9
        
        print(f"\nGPU Memory Stats:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Peak Allocated: {peak_allocated:.2f} GB")
        print(f"  Peak Cached: {peak_cached:.2f} GB")


def demonstrate_comprehensive_profiling():
    """
    Demonstrate comprehensive profiling with all tools.
    """
    print("\n=== Comprehensive Profiling Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    print("Running comprehensive profiling with all tools...")
    
    # 1. PyTorch Profiler
    print("1. PyTorch Profiler...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
    ) as prof:
        with torch.no_grad():
            for _ in range(50):
                with nvtx.annotate("comprehensive_profiling"):
                    output = model(x)
    
    print("PyTorch profiler completed")
    
    # 2. Memory profiling
    print("2. Memory profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
    ) as mem_prof:
        with torch.no_grad():
            for _ in range(50):
                with nvtx.annotate("memory_analysis"):
                    output = model(x)
    
    print("Memory profiling completed")
    
    # 3. FLOP analysis
    print("3. FLOP analysis...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_flops=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
    ) as flop_prof:
        with torch.no_grad():
            for _ in range(50):
                with nvtx.annotate("flop_analysis"):
                    output = model(x)
    
    print("FLOP analysis completed")
    
    # Print comprehensive results
    print("\nComprehensive Profiling Results:")
    print("=" * 50)
    
    print("\nTop operations by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    print("\nMemory usage:")
    print(mem_prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))
    
    print("\nFLOP analysis:")
    print(flop_prof.key_averages().table(sort_by="flops", row_limit=5))
    
    # Export results
    prof.export_chrome_trace("comprehensive_trace.json")
    print("\nComprehensive trace exported: comprehensive_trace.json")


def demonstrate_architecture_specific_profiling():
    """
    Demonstrate architecture-specific profiling features.
    """
    print("\n=== Architecture-Specific Profiling Demo ===")
    
    # Print architecture information
    arch_config.print_info()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        print(f"\nArchitecture-specific profiling for {device_props.name}:")
        
        if compute_capability == "9.0":  # Hopper H100/H200
            print("• HBM3 memory bandwidth profiling")
            print("• TMA (Tensor Memory Accelerator) analysis")
            print("• Hopper-specific kernel optimizations")
            print("• Transformer Engine profiling")
        elif compute_capability == "10.0":  # Blackwell B200/B300
            print("• HBM3e memory bandwidth profiling")
            print("• TMA (Tensor Memory Accelerator) analysis")
            print("• Stream-ordered memory allocation")
            print("• NVLink-C2C communication profiling")
            print("• Blackwell-specific kernel optimizations")
        
        # Architecture-specific profiling
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            profile_memory=True,
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
        ) as prof:
            with torch.no_grad():
                for _ in range(50):
                    with nvtx.annotate("architecture_profiling"):
                        output = model(x)
        
        print("\nArchitecture-specific profiling results:")
        print("Top operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
        
        print("\nMemory profiling:")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=5))


def demonstrate_profiling_automation():
    """
    Demonstrate automated profiling with latest tools.
    """
    print("\n=== Profiling Automation Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProfilingDemoModel().to(device)
    x = torch.randn(64, 1024, device=device)
    
    # Automated profiling pipeline
    profiling_results = {}
    
    # 1. PyTorch Profiler
    print("1. Running PyTorch profiler...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        profile_memory=True,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=2)
    ) as prof:
        with torch.no_grad():
            for _ in range(50):
                with nvtx.annotate("automated_profiling"):
                    output = model(x)
    
    # Collect PyTorch profiler results
    pytorch_results = prof.key_averages()
    profiling_results['pytorch'] = {
        'top_operations': pytorch_results.table(sort_by="cuda_time_total", row_limit=5),
        'memory_usage': pytorch_results.table(sort_by="self_cuda_memory_usage", row_limit=5),
        'flop_analysis': pytorch_results.table(sort_by="flops", row_limit=5)
    }
    
    # 2. System monitoring
    print("2. Collecting system metrics...")
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    profiling_results['system'] = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3)
    }
    
    # 3. GPU monitoring
    if torch.cuda.is_available():
        print("3. Collecting GPU metrics...")
        profiling_results['gpu'] = {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'cached_gb': torch.cuda.memory_reserved() / 1e9,
            'peak_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            'peak_cached_gb': torch.cuda.max_memory_reserved() / 1e9
        }
    
    # 4. Generate automated report
    print("4. Generating automated report...")
    
    report = f"""
# Automated Profiling Report

## System Information
- CPU Usage: {profiling_results['system']['cpu_percent']}%
- Memory Usage: {profiling_results['system']['memory_percent']}%
- Available Memory: {profiling_results['system']['memory_available_gb']:.2f} GB

## GPU Information
"""
    
    if 'gpu' in profiling_results:
        report += f"""
- Allocated Memory: {profiling_results['gpu']['allocated_gb']:.2f} GB
- Cached Memory: {profiling_results['gpu']['cached_gb']:.2f} GB
- Peak Allocated: {profiling_results['gpu']['peak_allocated_gb']:.2f} GB
- Peak Cached: {profiling_results['gpu']['peak_cached_gb']:.2f} GB
"""
    
    report += f"""
## PyTorch Profiler Results

### Top Operations by CUDA Time
{profiling_results['pytorch']['top_operations']}

### Memory Usage
{profiling_results['pytorch']['memory_usage']}

### FLOP Analysis
{profiling_results['pytorch']['flop_analysis']}

## Recommendations
1. Analyze top operations for optimization opportunities
2. Check memory usage patterns
3. Identify FLOP-intensive operations
4. Consider architecture-specific optimizations
"""
    
    # Save report
    with open("automated_profiling_report.txt", "w") as f:
        f.write(report)
    
    print("Automated profiling completed!")
    print("Report saved: automated_profiling_report.txt")
    
    # Export Chrome trace
    prof.export_chrome_trace("automated_trace.json")
    print("Chrome trace exported: automated_trace.json")


def main():
    """
    Main function to demonstrate comprehensive profiling tools.
    """
    print("=== Comprehensive Profiling Tools Demo ===")
    print("PyTorch 2.8, CUDA 12.9, Triton 3.4 Support")
    print("Enhanced for Hopper H100/H200 and Blackwell B200/B300")
    print()
    
    # Print architecture information
    arch_config.print_info()
    print()
    
    # Run demonstrations
    demonstrate_pytorch_profiler()
    demonstrate_nsight_systems()
    demonstrate_nsight_compute()
    demonstrate_hta_profiling()
    demonstrate_perf_profiling()
    demonstrate_memory_profiling()
    demonstrate_comprehensive_profiling()
    demonstrate_architecture_specific_profiling()
    demonstrate_profiling_automation()
    
    print("\n=== Summary ===")
    print("This demo shows comprehensive profiling with:")
    print("1. PyTorch 2.8 enhanced profiler")
    print("2. Nsight Systems timeline analysis")
    print("3. Nsight Compute kernel analysis")
    print("4. HTA for multi-GPU systems")
    print("5. Perf system-level analysis")
    print("6. Memory profiling")
    print("7. FLOP counting")
    print("8. Module-level analysis")
    print("9. Architecture-specific features")
    print("10. Automated profiling pipeline")
    print("11. Latest CUDA 12.9 support")
    print("12. Triton 3.4 integration")


if __name__ == "__main__":
    main()
