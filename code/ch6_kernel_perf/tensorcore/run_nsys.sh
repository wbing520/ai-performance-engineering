#!/usr/bin/env bash
nsys profile \
  --stats=true \
  --output=nsys_tensorcore_report \
  ./matmul_tensorcore_fp16