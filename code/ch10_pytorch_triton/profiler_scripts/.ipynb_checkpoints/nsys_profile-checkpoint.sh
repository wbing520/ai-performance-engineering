#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/script.py"
  exit 1
fi
SCRIPT=$1
nsys profile --force-overwrite=true -o $(basename $SCRIPT)_nsys python3 $SCRIPT
