#!/usr/bin/env bash
# Profile after_overlap_ddp.py with Nsight Systems

OUT="nsys_ddp"
CMD="python3 after_overlap_ddp.py --world_size 2"

rm -f ${OUT}.*  
nsys profile \
  --sample none \
  --trace=cuda,nccl \
  --output ${OUT} \
  ${CMD}
echo "Nsight Systems report: ${OUT}.qdrep"
