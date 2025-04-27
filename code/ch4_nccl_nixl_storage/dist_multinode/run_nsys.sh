#!/usr/bin/env bash
# Profile NCCL multi-node all-reduce with Nsight Systems
# Usage: note the PID, then run this script

OUT="nsys_multinode_nccl"
# Replace <pid> below with the NCCL script PID
nsys attach --pid <pid> \
  --trace=cuda,nccl \
  --output ${OUT}
echo "Nsight Systems trace saved as ${OUT}.qdrep"
