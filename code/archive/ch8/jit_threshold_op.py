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
# jit_threshold_op.py
# Chapter 8: PyTorch compiled version using torch.compile

import torch
import time

# PyTorch 2.8 compiled function
@torch.compile(fullgraph=True)
def threshold_op(X):
    return torch.maximum(X, torch.zeros_like(X))

def main():
    # Use updated PyTorch 2.8 features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    N = 1024 * 1024
    
    # Create input tensor (half positive, half negative for maximum divergence test)
    X = torch.empty(N, device=device)
    for i in range(N):
        X[i] = 1.0 if i % 2 == 0 else -1.0
    
    # Warm up compilation
    Y = threshold_op(X)
    torch.cuda.synchronize()
    
    # Time the compiled operation
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    Y = threshold_op(X)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"PyTorch compiled threshold operation time: {elapsed_time:.4f} ms")
    
    # Verify results
    expected = torch.where(X > 0, X, torch.zeros_like(X))
    correct = torch.allclose(Y, expected)
    print(f"Results: {'PASS' if correct else 'FAIL'}")

if __name__ == "__main__":
    main()
