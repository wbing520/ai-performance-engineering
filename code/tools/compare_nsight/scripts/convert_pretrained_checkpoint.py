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
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.utils import CLI, incremental_save


@torch.inference_mode()
def convert_checkpoint(checkpoint_file: Path, tokenizer_dir: Path, config_name: str, output_dir: Path) -> None:
    """Convert a checkpoint after pretraining.

    The pretrained checkpoint contains optimizer states and several other metadata that are not needed after training
    is finished. This script will export the state-dict of the model and place it in the chosen output folder together
    with the tokenizer and model config, which then can be loaded by other scripts for inference, evaluation, etc.

    Args:
        checkpoint_file: Path to a checkpoint file scripts produced by the scripts in ``lit_gpt/pretrain/``.
        tokenizer_dir: A path to the folder that holds the tokenizer configuration files that were used to train
            the model. All files with a name starting with 'tokenizer' will be copied to the output folder.
        config_name: The name of the model loaded with the ``lit_gpt.Config``. The configuration will be saved as a
            JSON file to the output folder.
        output_dir: The output folder where model state-dict file, the tokenizer config file, and the model config
            file will be saved.
    """

    if output_dir.is_dir() and output_dir.glob("*"):
        raise FileExistsError(
            f"The output folder exists and is not empty: {str(output_dir)}."
            " Please delete it first or choose a different name."
        )
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"The tokenizer_dir must be a directory: {str(output_dir)}.")

    output_dir.mkdir(parents=True)
    output_checkpoint_file = output_dir / "lit_model.pth"
    output_config_file = output_dir / "lit_config.json"

    # Save the config to output folder
    config = Config.from_name(config_name)
    with open(output_config_file, "w") as json_config:
        json.dump(asdict(config), json_config)

    # Export the tokenizer configuration to output folder
    for tokenizer_file in tokenizer_dir.glob("tokenizer*"):
        shutil.copyfile(tokenizer_file, output_dir / tokenizer_file.name)

    # Copy config for tokenization if found
    if (tokenizer_dir / "generation_config.json").is_file():
        shutil.copyfile(tokenizer_dir / "generation_config.json", output_dir / "generation_config.json")

    # Extract the model state dict and save to output folder
    with incremental_save(output_checkpoint_file) as saver:
        print("Processing", checkpoint_file)
        full_checkpoint = torch.load(str(checkpoint_file), mmap=True)
        loaded_state_dict = full_checkpoint["model"]
        converted_state_dict = {}
        for param_name, param in loaded_state_dict.items():
            saver.store_early(param)
            # remove prefix for compiled model (if any)
            param_name = param_name.replace("_orig_mod.", "")
            converted_state_dict[param_name] = param
        print(f"Saving converted checkpoint to {str(output_checkpoint_file)}.")
        saver.save(converted_state_dict)


if __name__ == "__main__":
    CLI(convert_checkpoint)

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
