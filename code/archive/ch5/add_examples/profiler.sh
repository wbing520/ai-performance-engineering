#!/usr/bin/env bash
make clean && make

# Nsight Systems timeline
nsys profile -o add_seq --force-overwrite ./addSequential
nsys profile -o add_par --force-overwrite ./addParallel

# Nsight Compute per-kernel metrics
ncu --target-processes all --set full -o add_seq_ncu ./addSequential
ncu --target-processes all --set full -o add_par_ncu ./addParallel

# PyTorch profiler
python3 add_parallel.py
