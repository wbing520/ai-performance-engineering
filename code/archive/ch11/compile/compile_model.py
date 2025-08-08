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
import torch._dynamo.config
import torch._inductor.config
import nvtx

# Configure for PyTorch 2.8 nightly and Triton 3.4
torch._dynamo.config.automatic_dynamic_shapes = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.cudagraphs = True
torch._inductor.config.triton.autotune_mode = "max-autotune"

# PyTorch 2.8 specific optimizations (removed invalid configs)
torch._inductor.config.triton.use_cudagraphs = True
torch._inductor.config.triton.max_autotune = True
torch._inductor.config.triton.debug = False

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

def train():
    # NVTX marker for profiling
    nvtx.range_push("train_step")
    model = SimpleModel().cuda()
    
    # Use torch.compile with latest optimizations
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(64, 1024).cuda()
    target = torch.randn(64, 1024).cuda()
    
    for _ in range(100):
        optimizer.zero_grad()
        nvtx.range_push("forward")
        output = model(data)
        nvtx.range_pop()
        loss = nn.MSELoss()(output, target)
        nvtx.range_push("backward")
        loss.backward()
        nvtx.range_pop()
        optimizer.step()
    nvtx.range_pop()

if __name__ == "__main__":
    train()
