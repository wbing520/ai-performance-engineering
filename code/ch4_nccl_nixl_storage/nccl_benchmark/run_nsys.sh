#!/usr/bin/env bash
# Profile peer-to-peer benchmark with Nsight Systems
nsys profile --sample none --trace=cuda --output nsys_p2p ./p2p_bandwidth_bench
