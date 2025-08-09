# before_no_overlap.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import sys

# A simple model with multiple layers to produce multiple gradients
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
    # Check if we have enough GPUs for distributed training
    if torch.cuda.device_count() < world_size:
        print(f"Warning: Only {torch.cuda.device_count()} GPU(s) available, but {world_size} requested.", flush=True)
        print("Falling back to single-GPU training.", flush=True)
        # Single GPU training
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MultiLayerNet(data.size(1)).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Single-GPU training completed. Loss: {loss.item():.4f}", flush=True)
        return
    
    # Multi-GPU distributed training
    dist.init_process_group("nccl", init_method="tcp://127.0.0.1:34567",
                           world_size=world_size, rank=rank)
    
    torch.cuda.set_device(rank)
    model = MultiLayerNet(data.size(1)).cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Move data to device
    data = data.cuda(rank)
    target = target.cuda(rank)
    
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
    print(f"Starting training with {torch.cuda.device_count()} GPU(s)", flush=True)
    world_size = min(2, torch.cuda.device_count())
    inp = torch.randn(128, 1024)
    tgt = torch.randn(128, 1)
    
    if world_size == 1:
        # Single GPU training
        print("Running single-GPU training", flush=True)
        train_no_overlap(0, 1, inp, tgt)
    else:
        # Multi-GPU training
        print("Running multi-GPU training", flush=True)
        mp.spawn(train_no_overlap, args=(world_size, inp, tgt), nprocs=world_size)
