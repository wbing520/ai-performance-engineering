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
from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple
import plotly.graph_objects as go
import numpy as np
from copy import copy
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

from .request import Request, ChunkedContextRequest
from .plotting import PlotData
from .load_generator import LoadGenerator
from .batcher import Batcher

class Engine:
    max_batch_size: int
    queue: List[Request]
    current_batch: Dict[int, Request]
    current_time: float = 0.
    plot_data: PlotData
    batcher: Batcher
    load_generator: LoadGenerator

    def __init__(self, max_batch_size: int, load_generator: LoadGenerator, batcher: Batcher):
        self.max_batch_size = max_batch_size
        self.plot_data = PlotData(num_slots=max_batch_size, engine=self)
        self.load_generator = load_generator
        self.load_generator.engine = self
        self.queue = []
        self.current_batch = {}
        self.batcher = batcher
        self.batcher.engine = self

    def run(self, time_limit: float=10.):
        while self.current_time < time_limit:
            # generate tokens
            for req in self.current_batch.values():
                req.tick()
            
            self.plot_data.track_previous_batch()

            # Complete previous requests. Keep only the requests, that are not completed yet
            self.current_batch = {
                slot: req for slot, req in self.current_batch.items()
                if req.tokens_generated < req.target_output_len_tokens
            }                    
            # generate load, add requests to the queue
            self.load_generator.generate_load()

            # Take requests from the queue to the batch
            self.batcher.add_requests()

            self.plot_data.track_current_batch()
            
            duration = self.get_current_batch_duration()
            self.current_time += duration

    def get_all_slots(self) -> Set[int]:
        return set(range(self.max_batch_size))

    def get_occupied_slots(self) -> Set[int]:
        return set(self.current_batch.keys())

    def assign_request_to_slot(self, req: Request, slot: int):
        req.started_at = self.current_time
        self.current_batch[slot] = req

    def add_requests_ifb(self):
        empty_slots = self.get_all_slots() - self.get_occupied_slots()
        for slot in empty_slots:
            if not len(self.queue):
                break
            req = self.queue.pop(0)
            self.assign_request_to_slot(req, slot)
        

    def get_prefilling_requests(self) -> List[Request]:
        return [req for req in self.current_batch.values() if req.is_in_prefill()]
    
    def get_decoding_requests(self) -> List[Request]:
        return [req for req in self.current_batch.values() if not req.is_in_prefill()]
    
    def get_current_batch_duration(self) -> float:
        decoding_requests = self.get_decoding_requests()
        # for no chunking the line below is equivalent to
        # prefill_time = sum([req.prefill_time for req in self.get_prefilling_requests()])
        prefill_time = sum([req.get_current_duration() for req in self.get_prefilling_requests()])
        itl_time = max([req.itl for req in decoding_requests]) if decoding_requests else 0
        return max(prefill_time, itl_time, 1.) # 1. is the minimal step duration


    
    

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
