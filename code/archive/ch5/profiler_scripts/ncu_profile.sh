#!/usr/bin/env bash
if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/executable"
  exit 1
fi
EXEC=$1
ncu --target-processes all --set full -o $(basename $EXEC)_ncu ${EXEC}
