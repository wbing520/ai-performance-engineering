#!/usr/bin/env bash
ncu --set full \
  --target-processes all \
  --output ncu_tensorcore_report \
  ./matmul_tensorcore_fp16