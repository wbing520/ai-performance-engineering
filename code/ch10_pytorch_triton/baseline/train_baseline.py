import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.nvtx as nvtx
from torch.utils.data import DataLoader, TensorDataset

# Simple placeholder model and data
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

def train():
    # NVTX marker for profiling
    nvtx.range_push("train_step")
    model = SimpleModel().cuda()
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
