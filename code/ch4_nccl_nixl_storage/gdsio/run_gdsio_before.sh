#!/usr/bin/env bash
# Run gdsio for CPU-mediated storage path
GDSIO=/usr/local/cuda-13.0/gds/tools/gdsio
FILE=/mnt/data/large_file
echo "Running CPU path..."
$GDSIO -f $FILE -d 0 -w 4 -s 10G -i 1M -I 0 -x 1
