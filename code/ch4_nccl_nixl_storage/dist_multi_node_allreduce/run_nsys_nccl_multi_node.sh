#!/usr/bin/env bash
# Profile NCCL all-reduce with Nsight Systems (GPU-direct RDMA)
export MASTER_ADDR=${MASTER_ADDR:-node0.example.com}
export MASTER_PORT=${MASTER_PORT:-29502}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

nsys profile --trace=cuda,nccl --sample none --output nsys_nccl_allreduce_multi_node   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=${NODE_RANK} allreduce_nccl_multi_node.py
