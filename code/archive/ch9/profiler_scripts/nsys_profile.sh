#!/usr/bin/env bash
if [ $# -ne 1 ]; then echo "Usage: $0 exec"; exit 1; fi
nsys profile --force-overwrite=true -o $(basename $1)_nsys $1
