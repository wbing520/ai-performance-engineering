#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1
ncu --target-processes=python3 --set full -o $(basename $SCRIPT)_ncu python3 $SCRIPT
