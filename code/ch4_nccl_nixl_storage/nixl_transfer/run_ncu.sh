#!/usr/bin/env bash
# Nsight Compute profiling
ncu --set full --launch-skip 0 --launch-count 1 --output ncu_nixl_report ./nixl_transfer
