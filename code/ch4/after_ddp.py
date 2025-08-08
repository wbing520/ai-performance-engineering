# after_ddp.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def train_ddp(rank, world_size):
    dist.init_process_group("nccl",
                           init_method="env://",
                           world_size=world_size, rank=rank)
    
    torch.cuda.set_device(rank)
    
    model = SimpleNet(input_size=1024, hidden_size=256)
    model.cuda(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Each process gets its own portion of data
    batch_size = 256
    data = torch.randn(batch_size, 1024).cuda(rank)
    target = torch.randn(batch_size, 1).cuda(rank)
    
    # Run one training iteration and measure time (on rank 0)
    torch.cuda.synchronize()
    if rank == 0:
        start = time.time()
    
    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    if rank == 0:
        elapsed = time.time() - start
        print(f"DDP step took {elapsed*1000:.2f} ms")
    
    dist.destroy_process_group()

def main():
    # Set up environment for distributed training
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    
    world_size = min(2, torch.cuda.device_count())
    if world_size < 2:
        print("This example requires at least 2 GPUs")
        return
    
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
