#!/usr/bin/env python3
"""
after_overlap_ddp.py
DistributedDataParallel with overlap example in PyTorch.

Hardware: Grace-Blackwell (sm_90) or Hopper (sm_80)
CUDA: 13.0
Python: 3.11, PyTorch nightly 2.8.0+cu13
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2,
                        help='number of processes (GPUs)')
    return parser.parse_args()

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

def train_ddp(rank, world_size):
    # Initialize NCCL process group
    dist.init_process_group("nccl",
                            init_method="tcp://127.0.0.1:29501",
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(rank)

    # Model, wrap in DDP
    model = MultiLayerNet(1024).cuda(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Dummy data per rank
    batch_size = 128
    data = torch.randn(batch_size, 1024, device=rank)
    target = torch.randn(batch_size, 1,   device=rank)

    # Warm-up
    for _ in range(5):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Timed iteration
    torch.cuda.synchronize(rank)
    start = time.time()

    optimizer.zero_grad()
    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()   # NCCL all-reduce happens concurrently
    optimizer.step()

    torch.cuda.synchronize(rank)
    elapsed_ms = (time.time() - start) * 1000
    if rank == 0:
        print(f"Rank {rank}: iteration took {elapsed_ms:.2f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    mp.spawn(train_ddp,
             args=(args.world_size,),
             nprocs=args.world_size,
             join=True)
