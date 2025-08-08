#!/usr/bin/env python3
"""
Architecture switching configuration for AI Performance Engineering.
Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100).
Updated for PyTorch 2.8, CUDA 12.9, and Triton 3.4.
"""

import torch
import os
from typing import Dict, Any, Optional

class ArchitectureConfig:
    """Configuration for different GPU architectures."""
    
    def __init__(self):
        self.arch = self._detect_architecture()
        self.config = self._get_architecture_config()
    
    def _detect_architecture(self) -> str:
        """Detect the current GPU architecture."""
        if not torch.cuda.is_available():
            return "cpu"
        
        device_props = torch.cuda.get_device_properties(0)
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        if compute_capability == "9.0":
            return "hopper"  # H100/H200
        elif compute_capability == "10.0":
            return "blackwell"  # B200/B300
        else:
            return "other"
    
    def _get_architecture_config(self) -> Dict[str, Any]:
        """Get configuration for the detected architecture."""
        configs = {
            "hopper": {
                "name": "Hopper H100/H200",
                "compute_capability": "9.0",
                "sm_version": "sm_90",
                "memory_bandwidth": "3.35 TB/s",
                "tensor_cores": "4th Gen",
                "features": ["HBM3", "Transformer Engine", "Dynamic Programming", "TMA"],
                "cuda_features": ["CUDA Graphs", "Dynamic Parallelism", "Unified Memory", "TMA", "HBM3"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "Dynamic shapes support",
                    "Mixed precision training",
                    "Transformer Engine",
                    "HBM3 optimizations"
                ],
                "triton_features": [
                    "Triton 3.4 optimizations",
                    "Hopper-specific kernels",
                    "HBM3 memory access patterns",
                    "TMA support"
                ],
                "profiling_tools": [
                    "Nsight Systems 2024.1",
                    "Nsight Compute 2024.1",
                    "HTA for multi-GPU",
                    "Enhanced PyTorch profiler",
                    "Perf system analysis"
                ]
            },
            "blackwell": {
                "name": "Blackwell B200/B300",
                "compute_capability": "10.0",
                "sm_version": "sm_100",
                "memory_bandwidth": "3.2 TB/s",
                "tensor_cores": "4th Gen",
                "features": ["HBM3e", "TMA", "NVLink-C2C", "Stream-ordered Memory"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e Optimizations", "NVLink-C2C"],
                "pytorch_optimizations": [
                    "Blackwell-specific optimizations",
                    "HBM3e memory optimizations",
                    "TMA support",
                    "Stream-ordered allocation",
                    "NVLink-C2C communication"
                ],
                "triton_features": [
                    "Triton 3.4 Blackwell optimizations",
                    "HBM3e memory access patterns",
                    "TMA kernel support",
                    "Stream-ordered memory",
                    "Blackwell-specific kernels"
                ],
                "profiling_tools": [
                    "Nsight Systems 2024.1",
                    "Nsight Compute 2024.1",
                    "HTA for multi-GPU",
                    "Enhanced PyTorch profiler",
                    "Perf system analysis",
                    "Blackwell-specific metrics"
                ]
            },
            "other": {
                "name": "Other",
                "compute_capability": "Unknown",
                "sm_version": "Unknown",
                "memory_bandwidth": "Unknown",
                "tensor_cores": "Unknown",
                "features": [],
                "cuda_features": [],
                "pytorch_optimizations": [],
                "triton_features": [],
                "profiling_tools": []
            }
        }
        
        return configs.get(self.arch, configs["other"])
    
    def get_sm_version(self) -> str:
        """Get the SM version for compilation."""
        return self.config["sm_version"]
    
    def get_architecture_name(self) -> str:
        """Get the architecture name."""
        return self.config["name"]
    
    def get_features(self) -> list:
        """Get architecture-specific features."""
        return self.config["features"]
    
    def get_cuda_features(self) -> list:
        """Get CUDA features for this architecture."""
        return self.config["cuda_features"]
    
    def get_pytorch_optimizations(self) -> list:
        """Get PyTorch optimizations for this architecture."""
        return self.config["pytorch_optimizations"]
    
    def get_triton_features(self) -> list:
        """Get Triton features for this architecture."""
        return self.config["triton_features"]
    
    def get_profiling_tools(self) -> list:
        """Get available profiling tools for this architecture."""
        return self.config["profiling_tools"]
    
    def configure_pytorch_optimizations(self):
        """Configure PyTorch optimizations for the current architecture."""
        if not torch.cuda.is_available():
            return
        
        if self.arch == "hopper":
            # Hopper H100/H200 optimizations
            torch._inductor.config.triton.use_hopper_optimizations = True
            torch._inductor.config.triton.hbm3_optimizations = True
            torch._inductor.config.triton.tma_support = True
            torch._inductor.config.triton.transformer_engine = True
        elif self.arch == "blackwell":
            # Blackwell B200/B300 optimizations
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
    
    def print_info(self):
        """Print architecture information."""
        print(f"Architecture: {self.config['name']}")
        print(f"Compute Capability: {self.config['compute_capability']}")
        print(f"SM Version: {self.config['sm_version']}")
        print(f"Memory Bandwidth: {self.config['memory_bandwidth']}")
        print(f"Tensor Cores: {self.config['tensor_cores']}")
        print(f"Features: {', '.join(self.config['features'])}")
        print(f"CUDA Features: {', '.join(self.config['cuda_features'])}")
        print(f"PyTorch Optimizations: {', '.join(self.config['pytorch_optimizations'])}")
        print(f"Triton Features: {', '.join(self.config['triton_features'])}")
        print(f"Profiling Tools: {', '.join(self.config['profiling_tools'])}")

# Global instance
arch_config = ArchitectureConfig()

def get_architecture() -> str:
    """Get the current architecture."""
    return arch_config.arch

def get_sm_version() -> str:
    """Get the SM version for compilation."""
    return arch_config.get_sm_version()

def configure_optimizations():
    """Configure optimizations for the current architecture."""
    arch_config.configure_pytorch_optimizations()

def print_architecture_info():
    """Print current architecture information."""
    arch_config.print_info()

if __name__ == "__main__":
    print_architecture_info()
    configure_optimizations()
