#!/usr/bin/env python3
"""
Architecture switching configuration for AI Performance Engineering.
Supports Hopper H100/H200 (sm_90) and Blackwell B200/B300 (sm_100).
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
                "features": ["HBM3", "Transformer Engine", "Dynamic Programming"],
                "cuda_features": ["CUDA Graphs", "Dynamic Parallelism", "Unified Memory"],
                "pytorch_optimizations": [
                    "torch.compile with max-autotune",
                    "Dynamic shapes support",
                    "Mixed precision training"
                ]
            },
            "blackwell": {
                "name": "Blackwell B200/B300",
                "compute_capability": "10.0",
                "sm_version": "sm_100",
                "memory_bandwidth": "3.2 TB/s",
                "tensor_cores": "4th Gen",
                "features": ["HBM3e", "TMA", "NVLink-C2C"],
                "cuda_features": ["Stream-ordered Memory", "TMA", "HBM3e Optimizations"],
                "pytorch_optimizations": [
                    "Blackwell-specific optimizations",
                    "HBM3e memory optimizations",
                    "TMA support"
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
                "pytorch_optimizations": []
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

# Global instance
arch_config = ArchitectureConfig()

def get_architecture() -> str:
    """Get the current architecture."""
    return arch_config.arch

def get_sm_version() -> str:
    """Get the SM version for compilation."""
    return arch_config.get_sm_version()

def print_architecture_info():
    """Print current architecture information."""
    arch_config.print_info()

if __name__ == "__main__":
    print_architecture_info()
