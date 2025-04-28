#!/usr/bin/env bash
# Profile before_reinit_comm.py with Nsight Compute
ncu --target-processes all --set full --launch-skip 0 --launch-count 5 --output ncu_before_report   python before_reinit_comm.py
