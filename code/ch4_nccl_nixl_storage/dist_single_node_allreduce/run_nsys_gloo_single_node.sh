#!/usr/bin/env bash
# Profile Gloo all-reduce with Nsight Systems (CPU+TCP)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29502}

nsys profile     --trace=osrt,python     --sample cpu     --output nsys_gloo_allreduc_single_nodee   torchrun --nnodes=1 --nproc_per_node=2 allreduce_gloo_single_node.py
