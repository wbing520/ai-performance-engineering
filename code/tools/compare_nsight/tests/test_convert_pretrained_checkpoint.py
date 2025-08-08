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
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os

import torch


def test_convert_pretrained_checkpoint(tmp_path):
    from scripts.convert_pretrained_checkpoint import convert_checkpoint

    # Pretend we made a checkpoint from pretraining
    pretrained_checkpoint = {
        "model": {"some.module.weight": torch.rand(2, 2), "_orig_mod.some.other.module.weight": torch.rand(2, 2)},
        "the_optimizer": "optimizer_state",
        "other": 1,
    }
    torch.save(pretrained_checkpoint, tmp_path / "pretrained.pth")

    # Make a fake tokenizer config file
    llama_checkpoint_folder = tmp_path / "checkpoints" / "meta-llama" / "Llama-2-7b-hf"
    llama_checkpoint_folder.mkdir(parents=True)
    (llama_checkpoint_folder / "tokenizer_config.json").touch()

    convert_checkpoint(
        checkpoint_file=(tmp_path / "pretrained.pth"),
        tokenizer_dir=llama_checkpoint_folder,
        config_name="tiny-llama-1.1b",
        output_dir=(tmp_path / "converted"),
    )

    assert set(os.listdir(tmp_path / "converted")) == {"lit_model.pth", "lit_config.json", "tokenizer_config.json"}
    converted_checkpoint = torch.load(tmp_path / "converted" / "lit_model.pth")
    assert list(converted_checkpoint.keys()) == ["some.module.weight", "some.other.module.weight"]

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
