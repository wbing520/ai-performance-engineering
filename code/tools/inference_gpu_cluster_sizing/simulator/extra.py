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
from .metrics import Metrics
import numpy as np
import io
import sys
import difflib


def print_experiment_metrics(engine, show_median=False):
    print("# Experiment Config:")
    print(f"load_generator = {str(engine.load_generator)}")
    print(f"batcher = {str(engine.batcher)}")
    metrics: Metrics = engine.plot_data.metrics
    # we record the latency of every completed request
    e2e_latencies = metrics.get_e2e_latencies()
    ttfts = metrics.get_ttfts()
    itls = metrics.get_itls()

    print("\n# Latency Metrics:")
    print(f"Average E2E Latency: {np.mean(e2e_latencies):.2f}")
    print(f"Average TTFT: {np.mean(ttfts):.2f}")
    print(f"Average ITL: {np.mean(itls):.2f}")

    if show_median:
        print(f"Median E2E Latency: {np.percentile(e2e_latencies, 0.5):.2f}")
        print(f"Median TTFT: {np.percentile(ttfts, 0.5):.2f}")
        print(f"Median ITL: {np.percentile(itls, 0.5):.2f}")


    print("\n# Throughput Metrics:")
    num_requests: int = len(e2e_latencies)
    run_time: float = metrics.times[-1]

    requests_per_1k_ticks_per_instance: float = 1000.*num_requests/run_time

    print(f"Requests/(1K ticks)/instance = {requests_per_1k_ticks_per_instance:.2f}")

    current_batch_tokens = sum(req.tokens_generated for req in engine.current_batch.values())
    total_tokens_generated = sum(metrics.get_osls()) + current_batch_tokens
    tokens_per_1k_ticks_per_instance = 1000 * total_tokens_generated / run_time
    print(f"Tokens/(1K ticks)/instance = {tokens_per_1k_ticks_per_instance:.2f}")

def catpure_function_prints(fn):
    try:
        capturedOutput = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capturedOutput
        fn()
        sys.stdout = old_stdout
        return capturedOutput.getvalue()

    finally:
        if old_stdout:
            sys.stdout = old_stdout

def check_print_metrics(print_experiment_metrics_function, engine, show_median=False):
    test_print = catpure_function_prints(lambda: print_experiment_metrics_function(engine))
    valid_print = catpure_function_prints(lambda: print_experiment_metrics(engine, show_median))
    for l in difflib.unified_diff(test_print.split("\n"), valid_print.split("\n"), fromfile="Your Implementation", tofile="Reference"):
        print(l)
   
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
