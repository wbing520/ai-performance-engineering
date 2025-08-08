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
# before_dataparallel.py
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Dummy model and dataset
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def main():
    # Setup model and data
    input_size = 1024
    hidden_size = 256
    model = SimpleNet(input_size, hidden_size)
    
    # Check if we have multiple GPUs
    if torch.cuda.device_count() < 2:
        print("This example requires at least 2 GPUs")
        return
    
    model.cuda()  # move model to GPU 0, it will also replicate to GPU 1
    model = nn.DataParallel(model)  # utilize 2 GPUs (0 and 1 by default)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    data = torch.randn(512, input_size).cuda()  # batch of 512 on GPU0
    target = torch.randn(512, 1).cuda()  # target on GPU0
    
    # Timing a single training step
    torch.cuda.synchronize()
    start = time.time()
    
    output = model(data)  # forward (DP splits data internally)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()  # backward (DP gathers grads to GPU0)
    optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"DataParallel step took {elapsed*1000:.2f} ms")

if __name__ == "__main__":
    main()

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
