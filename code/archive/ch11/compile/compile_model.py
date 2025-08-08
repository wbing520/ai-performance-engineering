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
