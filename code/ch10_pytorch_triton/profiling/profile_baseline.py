import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity
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

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        for _ in range(10):
            optimizer.zero_grad()
            with record_function("forward"):
                output = model(data)
            loss = nn.MSELoss()(output, target)
            with record_function("backward"):
                loss.backward()
            optimizer.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    train()
