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
from typing import List, Optional, Dict, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import Engine

class Batcher:
    engine: 'Engine'

    def add_requests(self):
        engine = self.engine
        if engine.get_occupied_slots(): return # static batcher cannot batch together new prefills with old decodings
        for slot in engine.get_all_slots():
            if not len(engine.queue): # checking if we still have more requests to run
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)

    def __str__(self):
        return f"{self.__class__.__name__}"

class StaticBatcher(Batcher):
    pass

class IFBatcher(Batcher):
    engine: 'Engine'

    def add_requests(self):
        engine = self.engine
        empty_slots = engine.get_all_slots() - engine.get_occupied_slots()
        for slot in empty_slots:
            if not len(engine.queue):
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)

class IFBatcherWithOnePrefillOnly(IFBatcher):
    def add_requests(self):
        engine = self.engine
        # Only one request can be in prefill simultaneously
        if len(engine.get_prefilling_requests()):
            return
        empty_slots = engine.get_all_slots() - engine.get_occupied_slots()
        for slot in empty_slots:
            if not len(engine.queue):
                break
            request = engine.queue.pop(0)
            engine.assign_request_to_slot(request, slot)
            break # Only one new request can be taken
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
