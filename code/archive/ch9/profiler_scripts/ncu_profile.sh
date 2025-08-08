#!/usr/bin/env bash
if [ $# -ne 1 ]; then echo "Usage: $0 exec"; exit 1; fi
ncu --target-processes all --set full -o $(basename $1)_ncu $1
