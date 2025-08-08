#!/bin/bash

# Comprehensive Build Script for Architecture Switching
# Supports Hopper H100/H200 and Blackwell B200/B300

set -e

echo "=== AI Performance Engineering - Comprehensive Build ==="

# Function to detect current architecture
detect_architecture() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        if [[ "$gpu_name" == *"H100"* ]] || [[ "$gpu_name" == *"H200"* ]]; then
            echo "sm_90"
        elif [[ "$gpu_name" == *"B200"* ]] || [[ "$gpu_name" == *"B300"* ]]; then
            echo "sm_100"
        else
            echo "sm_90"
        fi
    else
        echo "sm_90"
    fi
}

# Detect current architecture
CURRENT_ARCH=$(detect_architecture)
echo "Detected architecture: $CURRENT_ARCH"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements_latest.txt

# Build all CUDA projects
echo "Building CUDA projects..."
find code -name "Makefile" -type f | while read -r makefile; do
    dir=$(dirname "$makefile")
    echo "Building $dir..."
    cd "$dir"
    make clean
    make ARCH=$CURRENT_ARCH
    cd - > /dev/null
done

# Run tests
echo "Running tests..."
python test_architecture_switching.py

echo "Build completed successfully!"
