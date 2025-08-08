#!/usr/bin/env bash
# Example: 2 processes on 1 node
export WORLD_SIZE=2
export MASTER_ADDR=localhost
export MASTER_PORT=29500
python3 -m torch.distributed.launch --nproc_per_node=2 ddp_training.py
