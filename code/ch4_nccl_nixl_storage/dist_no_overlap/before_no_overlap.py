#!/usr/bin/env python3
"""
before_no_overlap.py
Manual no-overlap gradient all-reduce example in PyTorch.

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

def train_no_overlap(rank, world_size):
    # Initialize process group using NCCL (but we'll do manual all_reduce)
    dist.init_process_group("nccl",
                            init_method="tcp://127.0.0.1:29500",
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(rank)

    # Model, optimizer, data
    model = MultiLayerNet(1024).cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(128, 1024, device=rank)
    target = torch.randn(128, 1, device=rank)

    # Warm-up
    for _ in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        # manual all-reduce
        for p in model.parameters():
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size
        optimizer.step()

    # Timed iteration
    torch.cuda.synchronize(rank)
    start = time.time()

    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()

    # Manual synchronous all-reduce **after** backward
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size
    optimizer.step()

    torch.cuda.synchronize(rank)
    elapsed_ms = (time.time() - start) * 1000
    if rank == 0:
        print(f"Rank {rank}: iteration took {elapsed_ms:.2f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    mp.spawn(train_no_overlap,
             args=(args.world_size,),
             nprocs=args.world_size,
             join=True)
