#!/usr/bin/env bash
# Profile coalesced_copy with Nsight Systems

NSYS_OUT="nsys_coalesced"
EXECUTABLE="./coalesced_copy"

# Remove old data
rm -f ${NSYS_OUT}.*  
nsys profile \
  --sample none \
  --trace=cuda,cudnn \
  --output ${NSYS_OUT} \
  ${EXECUTABLE}
echo "Nsight Systems trace saved as ${NSYS_OUT}.qdrep"
