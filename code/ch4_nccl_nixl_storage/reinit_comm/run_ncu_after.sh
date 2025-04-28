#!/usr/bin/env bash
# Profile after_reinit_comm.py with Nsight Compute
ncu --target-processes all --set full --launch-skip 0 --launch-count 5 --output ncu_after_report   python after_reinit_comm.py
