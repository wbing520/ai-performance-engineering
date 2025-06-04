import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.fc(x)

def train():
    setup()
    model = SimpleModel().cuda()
    ddp_model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    data = torch.randn(64, 1024).cuda()
    target = torch.randn(64, 1024).cuda()
    for _ in range(20):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    train()
