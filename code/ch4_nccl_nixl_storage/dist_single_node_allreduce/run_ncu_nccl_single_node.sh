#!/usr/bin/env bash
# Profile NCCL all-reduce kernels with Nsight Compute
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29502}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

ncu     --target-processes all     --set full     --launch-skip 0     --launch-count 1     --output ncu_nccl_allreduce_single_node_report   torchrun --nnodes=1 --nproc_per_node=2 allreduce_nccl_single_node.py
