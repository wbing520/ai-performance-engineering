#!/usr/bin/env bash
set -euo pipefail

compile() {
    local src=$1
    local out=$2
    nvcc -std=c++17 -O3 -arch=sm_100 -DCUDA_VERSION=12.8 -lnvtx3 "$src" -o "$out"
}

compile copyScalar.cu scalar
./scalar

compile copyVector.cu vector
./vector
