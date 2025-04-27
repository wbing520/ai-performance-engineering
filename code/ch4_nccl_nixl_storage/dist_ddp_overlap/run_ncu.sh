#!/usr/bin/env bash
# Profile after_overlap_ddp.py with Nsight Compute

OUT="ddp_ncu_report"
CMD="python3 after_overlap_ddp.py --world_size 2"

rm -f ${OUT}*  
ncu --target-processes all \
    --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --output ${OUT} \
    ${CMD}
echo "Nsight Compute report: ${OUT}.ncu-rep"
