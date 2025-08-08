#!/usr/bin/env bash
if [ $# -ne 1 ]; then echo "Usage: $0 path/to/executable"; exit 1; fi
EXEC=$1
nsys profile --force-overwrite -o $(basename $EXEC)_nsys ${EXEC}
