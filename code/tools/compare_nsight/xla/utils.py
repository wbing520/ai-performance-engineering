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

import itertools
from functools import partial
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
from lightning.fabric.strategies.xla_fsdp import XLAFSDPStrategy, _activation_checkpointing_auto_wrapper
from lightning_utilities.core.rank_zero import rank_prefixed_message

from lit_gpt import GPT


def rank_print(fabric: L.Fabric, message: object, *, flush: bool = True, **kwargs: Any) -> None:
    if fabric.local_rank == 0:
        message = str(message)
        # let each host print, but only on rank 0
        message = rank_prefixed_message(message, fabric.global_rank)
        # TPU VM will only print when the script finishes if `flush=False`
        print(message, flush=flush, **kwargs)


def materialize_parameters(module: torch.nn.Module, device: torch.device) -> None:
    for module_name, module in module.named_modules():
        if any(
            param.is_meta for param in itertools.chain(module.parameters(recurse=False), module.buffers(recurse=False))
        ):
            module.to_empty(device=device, recurse=False)
            module.reset_parameters()


def sequential_load_and_fsdp_wrap(
    fabric: L.Fabric, get_model: Callable[[], GPT], checkpoint_path: Path
) -> torch.nn.Module:
    assert fabric._launched
    # similar logic could be implemented for regular FSDP, but this implementation is specific to XLAFSDP
    assert isinstance(fabric.strategy, XLAFSDPStrategy)

    with fabric.init_module(empty_init=False), torch.device("meta"):
        model = get_model()

    # TODO: this could be made faster by broadcasting in separate process groups for each host
    if fabric.local_rank == 0:
        # load the full checkpoint on a single rank to limit the system memory usage
        state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=False)  # mmap=True hangs
    else:
        # XLA cannot broadcast different number of tensors or different shapes in each rank. To get around this
        # limitation, we need to load the checkpoint on meta device to get the correct number of tensors and materialize
        # them as necessary
        state_dict = torch.load(checkpoint_path, map_location="meta", mmap=False)

    fsdp_kwargs = fabric.strategy._parse_fsdp_kwargs()
    if "auto_wrapper_callable" in fsdp_kwargs:
        # includes activation checkpointing if configured
        wrap = fsdp_kwargs.pop("auto_wrapper_callable")
    else:
        wrap = partial(_activation_checkpointing_auto_wrapper, set())
    fsdp_kwargs.pop("auto_wrap_policy", None)  # this needs to be removed or else root wrapping would error

    for i, block in enumerate(model.transformer.h):
        rank_print(fabric, f"Broadcasting transformer block {i}")
        # get the relevant piece of the state dict
        to_load = {}
        for param_name, _ in block.named_parameters():
            if (key := f"transformer.h.{i}.{param_name}") not in state_dict:
                continue
            param = state_dict.pop(key)
            if not param.is_meta:
                to_load[param_name] = param
            else:
                # materialize this parameter for broadcast to work
                to_load[param_name] = torch.empty_like(param, device="cpu")

        to_load = fabric.broadcast(to_load)

        rank_print(fabric, f"Loading transformer block {i}")
        keys = block.load_state_dict(to_load, strict=False, assign=True)
        assert not keys.unexpected_keys

        # materialize any leftover meta parameters, regular FSDP does it automatically
        materialize_parameters(block, torch.device("cpu"))  # init on CPU, FSDP will shard and move it

        # XLA FSDP only supports fp32 parameters. If the checkpoint had a different dtype, this needs to be converted
        # since we are loading with assign=True
        block = block.to(torch.float32)

        # shard the block
        rank_print(fabric, f"Wrapping transformer block {i}")
        wrapped_block = wrap(block, **fsdp_kwargs)
        model.transformer.h[i] = wrapped_block

    # load the rest of the state_dict, this assumes that all keys need to be loaded
    # an alternative technique would be to do load the rest of the state dict at once, but we want to materialize
    # and move the params to the xla device to reduce the system memory usage
    for key in list(state_dict):
        rank_print(fabric, f"Loading {key}")
        param = state_dict.pop(key)
        if param.is_meta:
            # materialize this parameter for broadcast to work
            param = torch.empty_like(param, device="cpu")
        param = fabric.broadcast(param)
        param = param.to(device=fabric.device, dtype=torch.float32)
        keys = model.load_state_dict({key: param}, strict=False, assign=True)
        assert not keys.unexpected_keys
    assert not state_dict

    # materialize any leftover meta parameters, regular FSDP does it automatically
    rank_print(fabric, "Materializing leftover parameters")
    materialize_parameters(model, fabric.device)

    return model

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
