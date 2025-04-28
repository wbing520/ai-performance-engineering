#!/usr/bin/env bash
# Profile before_reinit_comm.py with Nsight Systems
nsys profile --trace=cuda,nccl --sample none --output nsys_before   python before_reinit_comm.py
