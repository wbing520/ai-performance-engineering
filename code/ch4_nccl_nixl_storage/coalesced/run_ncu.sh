#!/usr/bin/env bash
# Profile coalesced_copy with Nsight Compute

NCU_OUT="coalesced_ncu_report"
EXECUTABLE="./coalesced_copy"

# Remove old data
rm -f ${NCU_OUT}*  
ncu --target-processes all \
    --set full \
    --launch-skip 0 \
    --launch-count 1 \
    --output ${NCU_OUT} \
    ${EXECUTABLE}
echo "Nsight Compute report saved as ${NCU_OUT}.ncu-rep"
