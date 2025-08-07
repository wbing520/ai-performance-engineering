import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.nvtx as nvtx

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

def train():
    model = SimpleModel().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(64, 1024).cuda()
    target = torch.randn(64, 1024).cuda()

    # Latest PyTorch 2.8 profiler configuration
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, 
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        schedule=schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        )
    ) as prof:
        for _ in range(10):
            optimizer.zero_grad()
            with record_function("forward"):
                nvtx.range_push("forward")
                output = model(data)
                nvtx.range_pop()
            loss = nn.MSELoss()(output, target)
            with record_function("backward"):
                nvtx.range_push("backward")
                loss.backward()
                nvtx.range_pop()
            optimizer.step()
    
    print("PyTorch Profiler Results:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export to Chrome trace format for TensorBoard
    prof.export_chrome_trace("trace.json")
    print("Chrome trace exported to trace.json")

if __name__ == "__main__":
    train()
