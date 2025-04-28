#!/usr/bin/env bash
# Profile after_reinit_comm.py with Nsight Systems
nsys profile --trace=cuda,nccl --sample none --output nsys_after   python after_reinit_comm.py
