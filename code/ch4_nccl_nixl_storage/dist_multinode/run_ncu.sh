#!/usr/bin/env bash
# Profile NCCL multi-node all-reduce with Nsight Compute

OUT="multinode_ncu_report"
# Replace <pid> below with the NCCL script PID
ncu --target-processes all \
    --attach <pid> \
    --set full \
    --output ${OUT}
echo "Nsight Compute report saved as ${OUT}.ncu-rep"
