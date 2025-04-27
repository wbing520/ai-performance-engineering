#!/usr/bin/env bash
# Nsight Systems profiling
nsys profile --sample none --trace=cuda,cuda-hw --output nsys_nixl ./nixl_transfer
