#!/usr/bin/env bash
# Profile NCCL all-reduce with Nsight Systems (GPU-direct P2P/RDMA)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29502}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

nsys profile     --trace=cuda,nccl     --sample none     --output nsys_nccl_allreduce_single_node   torchrun --nnodes=1 --nproc_per_node=2 allreduce_nccl_single_node.py
