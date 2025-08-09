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
            "memory_bandwidth": "8.0 TB/s",
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
Chapter 2: AI System Hardware Overview
Hardware Information and Benchmarking

This example demonstrates hardware analysis and benchmarking:
- GPU architecture detection and capabilities
- Memory hierarchy analysis
- Blackwell B200/B300 specific features
- Performance benchmarking
- Latest profiling tools integration
"""

import torch
import psutil
import GPUtil
import time
import numpy as np
from typing import Dict, Any, List
import torch.cuda.nvtx as nvtx


def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device_props = torch.cuda.get_device_properties(0)
    
    # Calculate memory bandwidth (use default values for missing attributes)
    # Note: memoryClockRate and memoryBusWidth are not available in current PyTorch version
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Estimate memory bandwidth based on compute capability
    if device_props.major >= 10:  # Blackwell B200/B300
        memory_bandwidth = 8.0  # TB/s for HBM3e
    elif device_props.major == 9:  # Hopper H100/H200
        memory_bandwidth = 3.35  # TB/s for HBM3
    else:
        memory_bandwidth = 1.0  # Default GB/s for other architectures
    
    # Check for Blackwell B200/B300 features
    is_blackwell = device_props.major >= 10  # SM100 for Blackwell
    is_hopper = device_props.major == 9  # SM90 for Hopper
    
    return {
        "name": device_props.name,
        "compute_capability": compute_capability,
        "total_memory_gb": device_props.total_memory / 1e9,
        "memory_bandwidth_gbps": memory_bandwidth * 1000 if memory_bandwidth < 10 else memory_bandwidth * 1000,  # Convert to GB/s
        "max_threads_per_block": device_props.max_threads_per_multi_processor,
        "max_threads_per_sm": device_props.max_threads_per_multi_processor,
        "num_sms": device_props.multi_processor_count,
        "warp_size": device_props.warp_size,
        "max_shared_memory_per_block": device_props.shared_memory_per_block,
        "max_shared_memory_per_sm": device_props.shared_memory_per_multiprocessor,
        "l2_cache_size": device_props.L2_cache_size,
        "architecture": "Blackwell B200/B300" if is_blackwell else "Hopper H100/H200" if is_hopper else "Other",
        "hbm3e_memory": is_blackwell,  # Blackwell has HBM3e
        "hbm3_memory": is_hopper,  # Hopper has HBM3
        "memory_bandwidth_tbps": memory_bandwidth if is_blackwell or is_hopper else None,
        "tma_support": is_blackwell or is_hopper,  # Tensor Memory Accelerator (Hopper & Blackwell)
        "nvlink_c2c": is_blackwell,  # NVLink-C2C for direct GPU communication
        "tensor_cores": "5th Generation" if is_blackwell else "4th Generation" if is_hopper else "2nd Generation",
        "unified_memory": True,  # Grace-Blackwell unified memory
        "max_unified_memory_gb": 30 if is_blackwell else None,  # 30TB unified memory
    }


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
        "overall_percent": psutil.cpu_percent(interval=1),
    }
    
    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": memory.total / 1e9,
        "available_gb": memory.available / 1e9,
        "used_gb": memory.used / 1e9,
        "percent": memory.percent,
    }
    
    # Get GPU information
    gpu_info = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_info[f"gpu_{i}"] = {
                "name": gpu.name,
                "load": gpu.load,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "temperature": gpu.temperature,
                "uuid": gpu.uuid,
            }
    except:
        pass
    
    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpus": gpu_info,
    }


def demonstrate_blackwell_features():
    """Demonstrate Blackwell B200/B300 specific features."""
    print("\n=== Blackwell B200/B300 Features Demonstration ===\n")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell features")
        return
    
    gpu_info = get_gpu_info()
    
    if gpu_info.get("architecture") == "Blackwell B200/B300":
        print("✓ This is a Blackwell B200/B300 GPU")
        print(f"✓ Compute Capability: {gpu_info['compute_capability']} (SM100)")
        print(f"✓ Memory: {gpu_info['total_memory_gb']:.1f} GB HBM3e")
        print(f"✓ Memory Bandwidth: {gpu_info['memory_bandwidth_tbps']} TB/s")
        print("✓ 4th Generation Tensor Cores")
        print("✓ TMA (Tensor Memory Accelerator)")
        print("✓ NVLink-C2C (Direct GPU-to-GPU communication)")
        print("✓ Unified Memory Architecture")
        print(f"✓ Max Unified Memory: {gpu_info['max_unified_memory_gb']} TB")
    else:
        print("This is not a Blackwell B200/B300 GPU")
        print(f"GPU: {gpu_info['name']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']:.1f} GB/s")


def benchmark_memory_bandwidth():
    """Benchmark memory bandwidth to demonstrate HBM3e performance."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return
    
    print("\n=== Memory Bandwidth Benchmark ===")
    
    # Test different tensor sizes
    sizes = [1024, 2048, 4096, 8192, 16384]
    
    for size in sizes:
        try:
            # Create large tensors
            a = torch.randn(size, size, device='cuda')
            b = torch.randn(size, size, device='cuda')
            
            # Warmup
            for _ in range(5):
                _ = torch.mm(a, b)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Benchmark matrix multiplication
            with nvtx.range(f"gemm_{size}"):
                for _ in range(10):
                    c = torch.mm(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            bytes_transferred = 2 * size * size * 4  # 2 matrices * size^2 * 4 bytes per float
            bandwidth_gbps = (bytes_transferred / avg_time) / 1e9
            
            print(f"Size {size}x{size}: {avg_time:.4f}s, {bandwidth_gbps:.1f} GB/s")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Size {size}x{size}: OOM")
            else:
                print(f"Size {size}x{size}: Error")


def benchmark_tensor_operations():
    """Benchmark various tensor operations to demonstrate Blackwell optimizations."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tensor benchmark")
        return
    
    print("\n=== Tensor Operations Benchmark ===")
    
    # Test different operations
    operations = [
        ("Matrix Multiplication", lambda x, y: torch.mm(x, y)),
        ("Element-wise Addition", lambda x, y: x + y),
        ("Element-wise Multiplication", lambda x, y: x * y),
        ("Matrix Transpose", lambda x, y: x.t()),
        ("Reduction Sum", lambda x, y: x.sum()),
    ]
    
    size = 4096
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    for op_name, op_func in operations:
        try:
            # Warmup
            for _ in range(5):
                if op_name == "Reduction Sum":
                    _ = op_func(a, b)
                elif op_name == "Matrix Transpose":
                    _ = op_func(a, b)
                else:
                    _ = op_func(a, b)
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            # Benchmark
            with nvtx.range(f"tensor_op_{op_name.lower().replace(' ', '_')}"):
                for _ in range(50):
                    if op_name == "Reduction Sum":
                        result = op_func(a, b)
                    elif op_name == "Matrix Transpose":
                        result = op_func(a, b)
                    else:
                        result = op_func(a, b)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 50
            print(f"{op_name:25}: {avg_time:.6f}s")
            
        except Exception as e:
            print(f"{op_name:25}: Error - {e}")


def demonstrate_memory_hierarchy():
    """Demonstrate memory hierarchy analysis."""
    print("\n=== Memory Hierarchy Analysis ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory hierarchy")
        return
    
    gpu_info = get_gpu_info()
    
    print("Memory Hierarchy (from fastest to slowest):")
    print("1. Registers (per-thread)")
    print("2. Shared Memory (per-block)")
    print("3. L1 Cache (per-SM)")
    print("4. L2 Cache (global)")
    print("5. HBM3e Memory (global)")
    print("6. CPU Memory (via unified memory)")
    
    print(f"\nMemory Specifications:")
    print(f"• Shared Memory per Block: {gpu_info['max_shared_memory_per_block'] / 1024:.1f} KB")
    print(f"• Shared Memory per SM: {gpu_info['max_shared_memory_per_sm'] / 1024:.1f} KB")
    print(f"• L2 Cache Size: {gpu_info['l2_cache_size'] / 1024:.1f} KB")
    print(f"• Global Memory: {gpu_info['total_memory_gb']:.1f} GB")
    
    if gpu_info.get('hbm3e_memory'):
        print(f"• HBM3e Bandwidth: {gpu_info['memory_bandwidth_tbps']} TB/s")
        print("• Unified Memory: 30 TB total")


def demonstrate_profiling_capabilities():
    """Demonstrate the latest profiling capabilities."""
    print("\n=== Latest Profiling Capabilities ===")
    
    print("Available Profiling Tools:")
    print("1. Nsight Systems (nsys) - Timeline analysis")
    print("   Command: nsys profile -t cuda,nvtx,osrt -o timeline_profile python script.py")
    
    print("\n2. Nsight Compute (ncu) - Kernel-level analysis")
    print("   Command: ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile python script.py")
    
    print("\n3. PyTorch Profiler - Framework-level analysis")
    print("   Command: python -c \"import torch.profiler; print('Available')\"")
    
    print("\n4. HTA (Holistic Tracing Analysis) - Multi-GPU analysis")
    print("   Command: nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile python script.py")
    
    print("\n5. Perf - System-level analysis")
    print("   Command: perf record -g -p $(pgrep python) -o perf.data")
    
    print("\n6. Enhanced PyTorch Profiler - Memory, FLOPs, modules")
    print("   Features: record_shapes=True, with_stack=True, with_flops=True, profile_memory=True")
    
    # Test PyTorch profiler availability
    try:
        from torch.profiler import profile, ProfilerActivity
        print("\n✓ PyTorch Profiler is available")
    except ImportError:
        print("\n✗ PyTorch Profiler not available")


def demonstrate_system_monitoring():
    """Demonstrate comprehensive system monitoring."""
    print("\n=== System Monitoring Demo ===")
    
    system_info = get_system_info()
    
    print("CPU Information:")
    print(f"• Physical Cores: {system_info['cpu']['physical_cores']}")
    print(f"• Logical Cores: {system_info['cpu']['logical_cores']}")
    print(f"• Overall Usage: {system_info['cpu']['overall_percent']:.1f}%")
    
    print("\nMemory Information:")
    print(f"• Total Memory: {system_info['memory']['total_gb']:.1f} GB")
    print(f"• Available Memory: {system_info['memory']['available_gb']:.1f} GB")
    print(f"• Memory Usage: {system_info['memory']['percent']:.1f}%")
    
    if system_info.get('gpus'):
        print("\nGPU Information:")
        for gpu_id, gpu_data in system_info['gpus'].items():
            print(f"• {gpu_id}: {gpu_data['name']}")
            print(f"  - Utilization: {gpu_data['load'] * 100:.1f}%")
            print(f"  - Memory Used: {gpu_data['memory_used']} MB")
            print(f"  - Memory Total: {gpu_data['memory_total']} MB")
            print(f"  - Temperature: {gpu_data['temperature']}°C")
    
    # PyTorch GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"\nPyTorch GPU Memory:")
        print(f"• Allocated: {allocated:.2f} GB")
        print(f"• Cached: {cached:.2f} GB")


def demonstrate_blackwell_optimizations():
    """Demonstrate Blackwell B200/B300 specific optimizations."""
    print("\n=== Blackwell B200/B300 Optimizations ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell optimizations")
        return
    
    gpu_info = get_gpu_info()
    
    if gpu_info.get('architecture') == "Blackwell B200/B300":
        print("Blackwell B200/B300 Optimizations:")
        print("1. HBM3e Memory Optimizations")
        print("   • High-bandwidth memory (3.2 TB/s)")
        print("   • Optimized memory access patterns")
        print("   • Unified memory architecture")
        
        print("\n2. Tensor Core Optimizations")
        print("   • 4th Generation Tensor Cores")
        print("   • FP8/FP4 precision support")
        print("   • Enhanced matrix operations")
        
        print("\n3. TMA (Tensor Memory Accelerator)")
        print("   • Efficient data movement")
        print("   • Reduced memory latency")
        print("   • Optimized memory bandwidth")
        
        print("\n4. NVLink-C2C")
        print("   • Direct GPU-to-GPU communication")
        print("   • High-speed data transfer")
        print("   • Reduced communication overhead")
        
        print("\n5. Unified Memory")
        print("   • 30 TB total unified memory")
        print("   • Seamless CPU-GPU memory access")
        print("   • Optimized memory management")
    else:
        print("This GPU does not support Blackwell B200/B300 optimizations")


def main():
    """Main function to demonstrate hardware analysis."""
    print("=== Chapter 2: AI System Hardware Overview ===")
    print("Hardware Information and Benchmarking Demo")
    print("=" * 50)
    
    # Get hardware information
    gpu_info = get_gpu_info()
    system_info = get_system_info()
    
    print("\n1. GPU Information:")
    print("-" * 20)
    if "error" not in gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']:.1f} GB/s")
        print(f"Number of SMs: {gpu_info['num_sms']}")
        print(f"Max Threads per Block: {gpu_info['max_threads_per_block']}")
        print(f"Warp Size: {gpu_info['warp_size']}")
    else:
        print("CUDA not available")
    
    print("\n2. Blackwell B200/B300 Specific Information:")
    print("-" * 40)
    if "error" not in gpu_info:
        if gpu_info.get('architecture') == "Blackwell B200/B300":
            print("✓ This is a Blackwell B200/B300 GPU")
            print(f"✓ Compute Capability: {gpu_info['compute_capability']} (SM100)")
            print(f"✓ Memory: {gpu_info['total_memory_gb']:.1f} GB HBM3e")
            print(f"✓ Memory Bandwidth: {gpu_info['memory_bandwidth_tbps']} TB/s")
            print("✓ 4th Generation Tensor Cores")
            print("✓ TMA (Tensor Memory Accelerator)")
            print("✓ NVLink-C2C (Direct GPU-to-GPU communication)")
            print("✓ Unified Memory Architecture")
            print(f"✓ Max Unified Memory: {gpu_info['max_unified_memory_gb']} TB")
        else:
            print("This is not a Blackwell B200/B300 GPU")
            print(f"GPU: {gpu_info['name']}")
            print(f"Compute Capability: {gpu_info['compute_capability']}")
            print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']:.1f} GB/s")
    
    print("\n3. System Information:")
    print("-" * 20)
    print(f"CPU Cores: {system_info['cpu']['physical_cores']} physical, {system_info['cpu']['logical_cores']} logical")
    print(f"CPU Usage: {system_info['cpu']['overall_percent']:.1f}%")
    print(f"Memory: {system_info['memory']['total_gb']:.1f} GB total, {system_info['memory']['available_gb']:.1f} GB available")
    print(f"Memory Usage: {system_info['memory']['percent']:.1f}%")
    
    # Run demonstrations
    demonstrate_blackwell_features()
    benchmark_memory_bandwidth()
    benchmark_tensor_operations()
    demonstrate_memory_hierarchy()
    demonstrate_profiling_capabilities()
    demonstrate_system_monitoring()
    demonstrate_blackwell_optimizations()
    
    print("\n=== Summary ===")
    print("This demo shows hardware analysis capabilities:")
    print("1. GPU architecture detection and capabilities")
    print("2. Memory hierarchy analysis")
    print("3. Blackwell B200/B300 specific features")
    print("4. Performance benchmarking")
    print("5. System monitoring")
    print("6. Latest profiling tools integration")
    print("7. Memory bandwidth testing")
    print("8. Tensor operation benchmarking")
    print("9. Comprehensive system analysis")
    print("10. Blackwell-specific optimizations")


if __name__ == "__main__":
    main()

# Note: Architecture-specific optimizations are handled in arch_config.py
# The following configuration options are not available in the current PyTorch version:
# - torch._inductor.config.triton.use_hopper_optimizations
# - torch._inductor.config.triton.hbm3_optimizations
# - torch._inductor.config.triton.use_blackwell_optimizations
# - torch._inductor.config.triton.hbm3e_optimizations
# - torch._inductor.config.triton.tma_support
# - torch._inductor.config.triton.autotune_mode
# - torch._dynamo.config.automatic_dynamic_shapes
