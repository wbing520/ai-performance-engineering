#!/usr/bin/env bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all --output=ncu_add ./add
