import torch.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx
import torch
import os

def get_architecture():
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    
    # Architecture detection
    if compute_capability == "10.0":
        return "blackwell"  # B200/B300
    return "other"

def get_architecture_info():
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "8.0 TB/s",
            "tensor_cores": "5th Gen",
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
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md"), encoding="utf-8") as fo:
    readme = fo.read()

setup(
    name="lit-gpt",
    version="0.1.0",
    description="Open source large language model implementation",
    author="Lightning AI",
    url="https://github.com/lightning-AI/lit-gpt",
    install_requires=[
        "torch>=2.2.0",
        "lightning @ git+https://github.com/Lightning-AI/lightning@ed367ca675861cdf40dbad2e4d66f7eee2ec50af",
    ],
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)

# Architecture-specific optimizations
if torch.cuda.is_available():
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"

    triton_cfg = getattr(getattr(getattr(torch, "_inductor", None), "config", None), "triton", None)
    if triton_cfg is not None:
        if compute_capability == "10.0":
            if hasattr(triton_cfg, "use_blackwell_optimizations"):
                triton_cfg.use_blackwell_optimizations = True
            if hasattr(triton_cfg, "hbm3e_optimizations"):
                triton_cfg.hbm3e_optimizations = True
            if hasattr(triton_cfg, "tma_support"):
                triton_cfg.tma_support = True
        if hasattr(triton_cfg, "stream_ordered_memory"):
            triton_cfg.stream_ordered_memory = True
        if hasattr(triton_cfg, "unique_kernel_names"):
            triton_cfg.unique_kernel_names = True
        if hasattr(triton_cfg, "autotune_mode"):
            triton_cfg.autotune_mode = "max-autotune"

    dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_cfg is not None and hasattr(dynamo_cfg, "automatic_dynamic_shapes"):
        dynamo_cfg.automatic_dynamic_shapes = True
