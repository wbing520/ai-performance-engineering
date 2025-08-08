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
import torch
import torch.nn as nn
import torch.optim as optim
import time

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def main():
    device = torch.device("cuda")
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create input data
    data = torch.randn(64, 1024, device=device)
    target = torch.randn(64, 1024, device=device)

    # Warm-up run
    for _ in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

    # Regular execution timing
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    regular_time = (time.time() - start) * 1000

    # CUDA Graph execution
    model.train()
    optimizer.zero_grad()
    
    # Capture graph
    with torch.cuda.amp.autocast():
        with torch.cuda.stream(torch.cuda.Stream()):
            static_input = data.clone()
            static_target = target.clone()
            
            # Warm-up
            output = model(static_input)
            loss = nn.MSELoss()(output, static_target)
            loss.backward()
            optimizer.step()
            
            # Capture the graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                optimizer.zero_grad()
                output = model(static_input)
                loss = nn.MSELoss()(output, static_target)
                loss.backward()
                optimizer.step()

    # Execute graph
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        g.replay()
    
    torch.cuda.synchronize()
    graph_time = (time.time() - start) * 1000

    print("PyTorch CUDA Graph captured")
    print(f"Graph execution time: {graph_time:.1f} ms")
    print(f"Regular execution time: {regular_time:.1f} ms")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1024 / 1024:.0f} MB")
    print(f"Speedup: {regular_time / graph_time:.1f}x")

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
