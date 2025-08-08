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

import json
import os

import torch


def test_merge_lora(tmp_path, fake_checkpoint_dir):
    from lit_gpt.lora import GPT as LoRAGPT
    from lit_gpt.lora import lora_filter
    from lit_gpt.model import GPT
    from scripts.merge_lora import merge_lora

    # create fake data
    config = dict(block_size=128, padded_vocab_size=256, n_layer=3, n_head=8, n_embd=16)
    with open(fake_checkpoint_dir / "lit_config.json", "w") as fp:
        json.dump(config, fp)
    base_model = GPT.from_name("pythia-14m", **config)
    state_dict = base_model.state_dict()
    assert len(state_dict) == 40
    torch.save(state_dict, fake_checkpoint_dir / "lit_model.pth")
    lora_model = LoRAGPT.from_name("pythia-14m", **config, r=8, alpha=16, dropout=0.05, to_query=True, to_value=True)
    state_dict = {k: v for k, v in lora_model.state_dict().items() if lora_filter(k, v)}
    assert len(state_dict) == 6
    lora_path = tmp_path / "lora"
    torch.save(state_dict, lora_path)

    assert set(os.listdir(tmp_path)) == {"lora", "checkpoints"}
    merge_lora(lora_path, fake_checkpoint_dir, tmp_path)
    assert set(os.listdir(tmp_path)) == {"lora", "checkpoints", "lit_model.pth"}

    # assert that the merged weights can be loaded back into the base model
    merged = torch.load(tmp_path / "lit_model.pth")
    keys = base_model.load_state_dict(merged, strict=True)
    assert not keys.missing_keys
    assert not keys.unexpected_keys

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
