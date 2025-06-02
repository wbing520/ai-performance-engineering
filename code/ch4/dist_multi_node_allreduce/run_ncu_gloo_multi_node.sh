#!/usr/bin/env bash
# Profile Gloo all-reduce with Nsight Compute
export MASTER_ADDR=${MASTER_ADDR:-node0.example.com}
export MASTER_PORT=${MASTER_PORT:-29502}

ncu --target-processes all --set full --launch-skip 0 --launch-count 1 --output ncu_gloo_allreduce_multi_node_report   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=${NODE_RANK} allreduce_gloo_multi_node.py
