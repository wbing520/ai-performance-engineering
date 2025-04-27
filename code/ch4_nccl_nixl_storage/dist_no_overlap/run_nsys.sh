#!/usr/bin/env bash
# Profile before_no_overlap.py with Nsight Systems

OUT="nsys_nooverlap"
CMD="python3 before_no_overlap.py --world_size 2"

rm -f ${OUT}.*  
nsys profile \
  --sample none \
  --trace=cuda,nccl \
  --output ${OUT} \
  ${CMD}
echo "Nsight Systems report: ${OUT}.qdrep"
