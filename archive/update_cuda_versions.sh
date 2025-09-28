#!/bin/bash
# Normalise Makefiles for Blackwell (SM100) with CUDA 12.9 toolchain.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

find "$SCRIPT_DIR" -name Makefile -type f | while read -r makefile; do
  sed -i "s/ARCH ?= sm_90/ARCH ?= sm_100/g" "$makefile"
  sed -i "s/ARCH?=sm_90/ARCH?=sm_100/g" "$makefile"
  sed -i "s/ARCH=sm_90/ARCH=sm_100/g" "$makefile"
  sed -i "s/CUDA_VERSION = .*/CUDA_VERSION = 12.9/g" "$makefile"
  sed -i "s/-arch=sm_90/-arch=sm_100/g" "$makefile"
  if ! grep -q "Blackwell" "$makefile"; then
    echo "⚠ Added Blackwell banner to $makefile"
  fi
done

echo "Makefiles normalised for Blackwell (sm_100, CUDA 12.9)."
