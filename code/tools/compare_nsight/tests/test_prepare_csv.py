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
import subprocess
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import ANY, call


def test_prepare_csv(tmp_path, fake_checkpoint_dir):
    with mock.patch("lit_gpt.tokenizer.Tokenizer"):
        from scripts.prepare_csv import prepare

    # create fake data
    config = dict(block_size=128, padded_vocab_size=256, n_layer=3, n_head=8, n_embd=16)
    with open(fake_checkpoint_dir / "lit_config.json", "w") as fp:
        json.dump(config, fp)
    csv_path = tmp_path / "data.csv"
    mock_data = (
        "instruction,input,output\n"
        "Add,2+2,4\n"
        "Subtract,5-3,2\n"
        "Multiply,6*4,24\n"
        "Divide,10/2,5\n"
        "Exponentiate,2^3,8\n"
        "Square root,√9,3\n"
    )
    with open(csv_path, "w", encoding="utf-8") as fp:
        fp.write(mock_data)

    with mock.patch("torch.save") as save_mock:
        prepare(csv_path, destination_path=tmp_path, checkpoint_dir=fake_checkpoint_dir, test_split_fraction=0.5)

    assert len(save_mock.mock_calls) == 2
    train_calls, test_calls = save_mock.mock_calls
    assert train_calls == call(
        [
            {"instruction": "Add", "input": "2+2", "output": "4", "input_ids": ANY, "labels": ANY},
            {"instruction": "Divide", "input": "10/2", "output": "5", "input_ids": ANY, "labels": ANY},
            {"instruction": "Multiply", "input": "6*4", "output": "24", "input_ids": ANY, "labels": ANY},
        ],
        tmp_path / "train.pt",
    )
    assert test_calls == call(
        [
            {"instruction": "Exponentiate", "input": "2^3", "output": "8", "input_ids": ANY, "labels": ANY},
            {"instruction": "Subtract", "input": "5-3", "output": "2", "input_ids": ANY, "labels": ANY},
            {"instruction": "Square root", "input": "√9", "output": "3", "input_ids": ANY, "labels": ANY},
        ],
        tmp_path / "test.pt",
    )


def test_cli():
    cli_path = Path(__file__).parent.parent / "scripts" / "prepare_csv.py"
    output = subprocess.check_output([sys.executable, cli_path, "-h"])
    output = str(output.decode())
    assert "Prepare a CSV dataset" in output

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
