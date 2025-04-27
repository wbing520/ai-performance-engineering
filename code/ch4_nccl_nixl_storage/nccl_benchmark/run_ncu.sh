#!/usr/bin/env bash
# Profile peer-to-peer benchmark with Nsight Compute
ncu --set full --launch-skip 0 --launch-count 1 --output ncu_p2p_report ./p2p_bandwidth_bench
