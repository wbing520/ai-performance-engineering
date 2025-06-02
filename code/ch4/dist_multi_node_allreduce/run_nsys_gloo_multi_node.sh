#!/usr/bin/env bash
# Profile Gloo all-reduce with Nsight Systems (CPU+TCP)
export MASTER_ADDR=${MASTER_ADDR:-node0.example.com}
export MASTER_PORT=${MASTER_PORT:-29502}

nsys profile --trace=osrt,python --sample cpu --output nsys_gloo_allreduce_multi_node   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=${NODE_RANK} allreduce_gloo_multi_node.py
