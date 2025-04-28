#!/usr/bin/env bash
# Run gdsio for GPUDirect Storage path
GDSIO=/usr/local/cuda-13.0/gds/tools/gdsio
FILE=/mnt/data/large_file
echo "Running GPU path (GDS)..."
$GDSIO -f $FILE -d 0 -w 4 -s 10G -i 1M -I 0 -x 0
