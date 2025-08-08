import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

# Simple model with multiple layers to produce multiple gradients
class MultiLayerNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_no_overlap(rank, world_size, data, target):
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:34567",
                             world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    model = MultiLayerNet(data.size(1)).cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    # Backward pass (compute all gradients locally)
    loss.backward()
    # Manually all-reduce gradients after backward (synchronous communication)
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size  # average gradients
    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    # Dummy data (128 samples, feature dimension 1024)
    inp = torch.randn(128, 1024)
    tgt = torch.randn(128, 1)
    mp.spawn(train_no_overlap, args=(world_size, inp, tgt), nprocs=world_size)
