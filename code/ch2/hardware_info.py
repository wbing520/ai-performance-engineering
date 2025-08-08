#!/usr/bin/env python3
"""
Chapter 2: AI System Hardware Overview
Hardware Information and Monitoring

This example demonstrates the core concepts from Chapter 2:
- GPU architecture and memory hierarchy
- System monitoring and utilization
- Hardware capabilities and limitations
- Blackwell B200/B300 specific features
"""

import torch
import torch.cuda as cuda
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Tuple, Optional
import subprocess
import json
import time
import torch.cuda.nvtx as nvtx


class HardwareMonitor:
    """Monitor and display hardware information for AI systems."""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
    def get_gpu_architecture_info(self) -> Dict[str, any]:
        """Get detailed GPU architecture information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        gpu_info = {}
        
        for i in range(self.gpu_count):
            props = cuda.get_device_properties(i)
            gpu_info[f"gpu_{i}"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": props.total_memory / (1024**3),
                "multiprocessor_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block,
                "max_shared_memory_per_block": props.max_shared_memory_per_block,
                "warp_size": props.warp_size,
                "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
                "max_blocks_per_multiprocessor": props.max_blocks_per_multiprocessor,
                "memory_clock_rate_mhz": props.memory_clock_rate,
                "memory_bus_width": props.memory_bus_width,
                "l2_cache_size": props.l2_cache_size,
                "total_constant_memory": props.total_constant_memory,
                "texture_alignment": props.texture_alignment,
                "max_pitch": props.max_pitch,
                "max_grid_size": props.max_grid_size,
                "max_block_dim": props.max_block_dim,
                "compute_mode": props.compute_mode,
                "concurrent_kernels": props.concurrent_kernels,
                "ecc_enabled": props.ecc_enabled,
                "unified_addressing": props.unified_addressing,
                "managed_memory": props.managed_memory,
                "concurrent_managed_access": props.concurrent_managed_access,
                "direct_managed_memory_access": props.direct_managed_memory_access,
                "single_to_double_precision_performance_ratio": props.single_to_double_precision_performance_ratio,
                "pageable_memory_access": props.pageable_memory_access,
                "concurrent_host_access": props.concurrent_host_access,
            }
            
        return gpu_info
    
    def get_blackwell_specific_info(self) -> Dict[str, any]:
        """Get Blackwell B200/B300 specific information."""
        blackwell_info = {}
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        for i in range(self.gpu_count):
            props = cuda.get_device_properties(i)
            compute_cap = f"{props.major}.{props.minor}"
            
            # Check if this is a Blackwell GPU (SM100)
            is_blackwell = compute_cap == "10.0"
            
            blackwell_info[f"gpu_{i}"] = {
                "is_blackwell": is_blackwell,
                "compute_capability": compute_cap,
                "architecture": "Blackwell B200/B300" if is_blackwell else "Other",
                "hbm3e_memory": is_blackwell,  # Blackwell has HBM3e
                "tma_support": is_blackwell,    # Tensor Memory Accelerator
                "nvlink_c2c": is_blackwell,    # Direct GPU-to-GPU communication
                "memory_bandwidth_tbps": 3.2 if is_blackwell else None,  # HBM3e bandwidth
                "tensor_cores_gen": 4 if is_blackwell else None,  # 4th generation
                "max_memory_gb": 192 if is_blackwell else props.total_memory / (1024**3),  # B200 has 192GB
                "power_consumption_w": 800 if is_blackwell else None,  # B200 power
            }
            
        return blackwell_info
    
    def get_memory_hierarchy_info(self) -> Dict[str, any]:
        """Get information about the memory hierarchy."""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info["system"] = {
            "total_gb": system_memory.total / (1024**3),
            "available_gb": system_memory.available / (1024**3),
            "used_gb": system_memory.used / (1024**3),
            "percent_used": system_memory.percent,
        }
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(self.gpu_count):
                memory_info[f"gpu_{i}_memory"] = {
                    "total_gb": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                    "cached_gb": torch.cuda.memory_reserved(i) / (1024**3),
                    "free_gb": (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / (1024**3),
                }
                
        return memory_info
    
    def get_system_utilization(self) -> Dict[str, any]:
        """Get current system utilization metrics."""
        utilization = {}
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
        utilization["cpu"] = {
            "overall_percent": psutil.cpu_percent(interval=1),
            "per_core_percent": cpu_percent,
            "core_count": psutil.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
        }
        
        # Memory utilization
        memory = psutil.virtual_memory()
        utilization["memory"] = {
            "percent_used": memory.percent,
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
        }
        
        # GPU utilization
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                utilization["gpu"] = {}
                for i, gpu in enumerate(gpus):
                    utilization["gpu"][f"gpu_{i}"] = {
                        "utilization_percent": gpu.load * 100,
                        "memory_used_gb": gpu.memoryUsed,
                        "memory_total_gb": gpu.memoryTotal,
                        "temperature_celsius": gpu.temperature,
                    }
            except:
                utilization["gpu"] = {"error": "Could not get GPU utilization"}
                
        return utilization
    
    def get_nvlink_info(self) -> Dict[str, any]:
        """Get NVLink information if available."""
        nvlink_info = {}
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        try:
            # Try to get NVLink information using nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "nvlink", "--status"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                nvlink_info["status"] = "Available"
                nvlink_info["output"] = result.stdout
            else:
                nvlink_info["status"] = "Not available or error"
                nvlink_info["error"] = result.stderr
                
        except FileNotFoundError:
            nvlink_info["status"] = "nvidia-smi not found"
            
        return nvlink_info
    
    def get_tensor_core_info(self) -> Dict[str, any]:
        """Get information about Tensor Cores and compute capabilities."""
        tensor_info = {}
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        for i in range(self.gpu_count):
            props = cuda.get_device_properties(i)
            compute_cap = f"{props.major}.{props.minor}"
            
            # Determine Tensor Core support based on compute capability
            tensor_core_support = {
                "fp16": float(compute_cap) >= 7.0,  # Volta and later
                "fp8": float(compute_cap) >= 8.9,   # Hopper and later
                "fp4": float(compute_cap) >= 10.0,  # Blackwell and later
                "bf16": float(compute_cap) >= 8.0,  # Ampere and later
            }
            
            tensor_info[f"gpu_{i}"] = {
                "compute_capability": compute_cap,
                "tensor_core_support": tensor_core_support,
                "sm_count": props.multi_processor_count,
                "max_threads_per_sm": props.max_threads_per_multiprocessor,
            }
            
        return tensor_info
    
    def demonstrate_memory_hierarchy(self):
        """Demonstrate the memory hierarchy with a practical example."""
        print("=== Memory Hierarchy Demonstration ===\n")
        
        if not torch.cuda.is_available():
            print("CUDA not available - running CPU-only demonstration")
            return
            
        device = torch.device('cuda:0')
        
        # Demonstrate different memory types
        print("1. Creating tensors in different memory locations:")
        
        # CPU memory
        cpu_tensor = torch.randn(1000, 1000, device='cpu')
        print(f"   CPU tensor: {cpu_tensor.device}, size: {cpu_tensor.numel() * cpu_tensor.element_size() / 1024**2:.2f} MB")
        
        # GPU global memory
        gpu_tensor = torch.randn(1000, 1000, device=device)
        print(f"   GPU tensor: {gpu_tensor.device}, size: {gpu_tensor.numel() * gpu_tensor.element_size() / 1024**2:.2f} MB")
        
        # Demonstrate memory transfer
        print("\n2. Memory transfer demonstration:")
        start_time = time.time()
        gpu_tensor_from_cpu = cpu_tensor.to(device)
        transfer_time = time.time() - start_time
        print(f"   CPU to GPU transfer time: {transfer_time*1000:.2f} ms")
        
        # Demonstrate unified memory (if available)
        if torch.cuda.get_device_properties(0).unified_addressing:
            print("\n3. Unified memory demonstration:")
            unified_tensor = torch.randn(1000, 1000, device=device, pin_memory=True)
            print(f"   Unified memory tensor created: {unified_tensor.device}")
        
        # Memory usage summary
        print("\n4. Current memory usage:")
        memory_info = self.get_memory_hierarchy_info()
        for key, info in memory_info.items():
            if "gpu" in key:
                print(f"   {key}: {info['allocated_gb']:.2f} GB allocated, {info['cached_gb']:.2f} GB cached")
    
    def demonstrate_compute_capabilities(self):
        """Demonstrate different compute capabilities and precision formats."""
        print("\n=== Compute Capabilities Demonstration ===\n")
        
        if not torch.cuda.is_available():
            print("CUDA not available")
            return
            
        device = torch.device('cuda:0')
        props = cuda.get_device_properties(0)
        compute_cap = f"{props.major}.{props.minor}"
        
        print(f"GPU: {props.name}")
        print(f"Compute Capability: {compute_cap}")
        print(f"SM Count: {props.multi_processor_count}")
        print(f"Max Threads per SM: {props.max_threads_per_multiprocessor}")
        
        # Test different precision formats
        print("\n1. Precision format demonstration:")
        
        # FP32 (standard precision)
        fp32_tensor = torch.randn(1000, 1000, dtype=torch.float32, device=device)
        print(f"   FP32 tensor: {fp32_tensor.dtype}, size: {fp32_tensor.numel() * fp32_tensor.element_size() / 1024**2:.2f} MB")
        
        # FP16 (half precision) - if supported
        if float(compute_cap) >= 7.0:
            fp16_tensor = torch.randn(1000, 1000, dtype=torch.float16, device=device)
            print(f"   FP16 tensor: {fp16_tensor.dtype}, size: {fp16_tensor.numel() * fp16_tensor.element_size() / 1024**2:.2f} MB")
            
            # Demonstrate speedup
            start_time = time.time()
            _ = torch.mm(fp32_tensor, fp32_tensor)
            fp32_time = time.time() - start_time
            
            start_time = time.time()
            _ = torch.mm(fp16_tensor, fp16_tensor)
            fp16_time = time.time() - start_time
            
            if fp16_time > 0:
                speedup = fp32_time / fp16_time
                print(f"   FP16 speedup over FP32: {speedup:.2f}x")
        
        # BF16 (bfloat16) - if supported
        if float(compute_cap) >= 8.0:
            bf16_tensor = torch.randn(1000, 1000, dtype=torch.bfloat16, device=device)
            print(f"   BF16 tensor: {bf16_tensor.dtype}, size: {bf16_tensor.numel() * bf16_tensor.element_size() / 1024**2:.2f} MB")
        
        # FP8 (8-bit precision) - if supported
        if float(compute_cap) >= 8.9:
            # Note: PyTorch doesn't have native FP8 yet, but we can demonstrate the concept
            print(f"   FP8 support: Available (Hopper and later)")
        
        # FP4 (4-bit precision) - Blackwell specific
        if float(compute_cap) >= 10.0:
            print(f"   FP4 support: Available (Blackwell B200/B300)")
        
        # Demonstrate memory bandwidth
        print("\n2. Memory bandwidth test:")
        large_tensor = torch.randn(10000, 10000, device=device)
        start_time = time.time()
        _ = large_tensor + large_tensor
        bandwidth_time = time.time() - start_time
        
        memory_accessed = large_tensor.numel() * large_tensor.element_size() * 2  # Read + write
        bandwidth_gbps = (memory_accessed / bandwidth_time) / (1024**3)
        print(f"   Estimated memory bandwidth: {bandwidth_gbps:.2f} GB/s")
    
    def demonstrate_blackwell_features(self):
        """Demonstrate Blackwell B200/B300 specific features."""
        print("\n=== Blackwell B200/B300 Features Demonstration ===\n")
        
        if not torch.cuda.is_available():
            print("CUDA not available")
            return
            
        device = torch.device('cuda:0')
        props = cuda.get_device_properties(0)
        compute_cap = f"{props.major}.{props.minor}"
        
        print(f"GPU: {props.name}")
        print(f"Compute Capability: {compute_cap}")
        
        # Check if this is a Blackwell GPU
        is_blackwell = compute_cap == "10.0"
        
        if is_blackwell:
            print("✓ This is a Blackwell B200/B300 GPU")
            print("✓ SM100 Architecture (Compute Capability 10.0)")
            print("✓ HBM3e Memory (up to 3.2TB/s bandwidth)")
            print("✓ 4th Generation Tensor Cores")
            print("✓ TMA (Tensor Memory Accelerator)")
            print("✓ NVLink-C2C (Direct GPU-to-GPU communication)")
            
            # Test HBM3e memory bandwidth
            print("\nTesting HBM3e memory bandwidth...")
            sizes = [1024, 2048, 4096, 8192]
            
            for size in sizes:
                try:
                    a = torch.randn(size, size, device=device)
                    b = torch.randn(size, size, device=device)
                    
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    with nvtx.annotate(f"blackwell_gemm_{size}"):
                        c = torch.mm(a, b)
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    avg_time = end_time - start_time
                    flops = 2 * size * size * size
                    tflops = flops / (avg_time * 1e12)
                    
                    print(f"   Size {size}x{size}: {avg_time:.4f}s, {tflops:.2f} TFLOPS")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"   Size {size}x{size}: OOM")
                    else:
                        print(f"   Size {size}x{size}: Error")
        else:
            print("This is not a Blackwell GPU")
            print(f"Current compute capability: {compute_cap}")
            print("Blackwell GPUs have compute capability 10.0")
    
    def print_hardware_summary(self):
        """Print a comprehensive hardware summary."""
        print("=== AI System Hardware Overview ===\n")
        
        print("1. GPU Architecture Information:")
        print("-" * 40)
        gpu_info = self.get_gpu_architecture_info()
        
        if "error" in gpu_info:
            print(f"   {gpu_info['error']}")
        else:
            for gpu_id, info in gpu_info.items():
                print(f"   {gpu_id}: {info['name']}")
                print(f"     Compute Capability: {info['compute_capability']}")
                print(f"     Memory: {info['total_memory_gb']:.1f} GB")
                print(f"     SMs: {info['multiprocessor_count']}")
                print(f"     Warp Size: {info['warp_size']}")
        
        print("\n2. Blackwell B200/B300 Specific Information:")
        print("-" * 40)
        blackwell_info = self.get_blackwell_specific_info()
        
        if "error" not in blackwell_info:
            for gpu_id, info in blackwell_info.items():
                print(f"   {gpu_id}: {info['architecture']}")
                if info['is_blackwell']:
                    print(f"     ✓ HBM3e Memory: {info['memory_bandwidth_tbps']} TB/s")
                    print(f"     ✓ TMA Support: {info['tma_support']}")
                    print(f"     ✓ NVLink-C2C: {info['nvlink_c2c']}")
                    print(f"     ✓ Tensor Cores: {info['tensor_cores_gen']}th Generation")
                    print(f"     ✓ Max Memory: {info['max_memory_gb']} GB")
                    print(f"     ✓ Power: {info['power_consumption_w']}W")
                else:
                    print(f"     Compute Capability: {info['compute_capability']}")
        
        print("\n3. Memory Hierarchy:")
        print("-" * 40)
        memory_info = self.get_memory_hierarchy_info()
        
        print(f"   System Memory: {memory_info['system']['total_gb']:.1f} GB total")
        print(f"   System Memory Used: {memory_info['system']['percent_used']:.1f}%")
        
        for key, info in memory_info.items():
            if "gpu" in key and "memory" in key:
                print(f"   {key}: {info['total_gb']:.1f} GB total, {info['allocated_gb']:.1f} GB allocated")
        
        print("\n4. Current Utilization:")
        print("-" * 40)
        utilization = self.get_system_utilization()
        
        print(f"   CPU: {utilization['cpu']['overall_percent']:.1f}%")
        print(f"   Memory: {utilization['memory']['percent_used']:.1f}%")
        
        if "gpu" in utilization and "error" not in utilization["gpu"]:
            for gpu_id, gpu_info in utilization["gpu"].items():
                print(f"   {gpu_id}: {gpu_info['utilization_percent']:.1f}% utilization, {gpu_info['temperature_celsius']:.1f}°C")
        
        print("\n5. Tensor Core Support:")
        print("-" * 40)
        tensor_info = self.get_tensor_core_info()
        
        if "error" not in tensor_info:
            for gpu_id, info in tensor_info.items():
                print(f"   {gpu_id}: Compute Capability {info['compute_capability']}")
                for precision, supported in info['tensor_core_support'].items():
                    status = "✓" if supported else "✗"
                    print(f"     {precision.upper()}: {status}")
        
        print("\n6. NVLink Status:")
        print("-" * 40)
        nvlink_info = self.get_nvlink_info()
        print(f"   Status: {nvlink_info['status']}")


def demonstrate_hardware_concepts():
    """Demonstrate the key hardware concepts from Chapter 2."""
    print("=== Chapter 2: AI System Hardware Overview Demo ===\n")
    
    monitor = HardwareMonitor()
    
    # Print comprehensive hardware summary
    monitor.print_hardware_summary()
    
    # Demonstrate memory hierarchy
    monitor.demonstrate_memory_hierarchy()
    
    # Demonstrate compute capabilities
    monitor.demonstrate_compute_capabilities()
    
    # Demonstrate Blackwell features
    monitor.demonstrate_blackwell_features()
    
    print("\n=== Key Takeaways from Chapter 2 ===")
    print("1. Unified Memory Architecture: CPU and GPU share coherent memory space")
    print("2. Memory Hierarchy: Registers → Shared Memory → L1 Cache → L2 Cache → HBM3e")
    print("3. Tensor Cores: Specialized units for matrix operations with reduced precision")
    print("4. NVLink-C2C: Direct GPU-to-GPU communication in Blackwell")
    print("5. Superchip Design: Grace CPU + Blackwell GPU in unified package")
    print("6. NVL72: 72-GPU rack with unified memory and NVSwitch fabric")
    print("7. Liquid Cooling: Essential for high-density compute")
    print("8. Power Management: 130kW per rack requires careful planning")
    print("9. Blackwell B200/B300: SM100 architecture with HBM3e memory")
    print("10. TMA: Tensor Memory Accelerator for efficient data movement")


if __name__ == "__main__":
    demonstrate_hardware_concepts()
