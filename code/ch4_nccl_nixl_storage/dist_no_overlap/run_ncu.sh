#!/usr/bin/env bash
# Profile before_no_overlap.py with Nsight Compute

OUT="nooverlap_ncu_report"
CMD="python3 before_no_overlap.py --world_size 2"

rm -f ${OUT}*  
ncu --target-processes all \
    --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --output ${OUT} \
    ${CMD}
echo "Nsight Compute report: ${OUT}.ncu-rep"
