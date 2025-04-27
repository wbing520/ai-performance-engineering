#!/usr/bin/env bash
# Profile Gloo all-reduce with Nsight Compute (minimal GPU kernels)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29502}

ncu     --target-processes all     --set full     --launch-skip 0     --launch-count 1     --output ncu_gloo_allreduce_single_node_report   torchrun --nnodes=1 --nproc_per_node=2 allreduce_gloo_single_node.py
