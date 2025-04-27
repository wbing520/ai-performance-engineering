#!/usr/bin/env bash
# Profile NCCL all-reduce kernels with Nsight Compute
export MASTER_ADDR=${MASTER_ADDR:-node0.example.com}
export MASTER_PORT=${MASTER_PORT:-29502}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

ncu --target-processes all --set full --launch-skip 0 --launch-count 1 --output ncu_nccl_allreduce_multi_node_report   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=${NODE_RANK} allreduce_nccl_multi_node.py
