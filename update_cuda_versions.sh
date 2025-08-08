#!/bin/bash

# Update script for CUDA 12.4 and modern GPU architecture compliance
# This script updates all Makefiles to use the latest CUDA version and optimizations

echo "Updating CUDA versions and modern GPU optimizations..."

# Find all Makefiles and update them
find . -name "Makefile" -type f | while read -r makefile; do
    echo "Updating $makefile..."
    
    # Update CUDA architecture to support modern GPUs (Ampere, Ada, Hopper)
    sed -i.bak 's/-arch=sm_70/-arch=sm_80/g' "$makefile"
    sed -i.bak 's/-arch=sm_75/-arch=sm_80/g' "$makefile"
    sed -i.bak 's/-arch=sm_86/-arch=sm_80/g' "$makefile"
    sed -i.bak 's/-arch=sm_90/-arch=sm_80/g' "$makefile"
    
    # Add CUDA version definition if not present
    if ! grep -q "CUDA_VERSION" "$makefile"; then
        # Add after the first line
        sed -i.bak '1a\
CUDA_VERSION = 12.4' "$makefile"
    fi
    
    # Update nvcc flags to include CUDA version
    sed -i.bak 's/nvcc.*-std=c++17/nvcc -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION)/g' "$makefile"
    
    # Add modern GPU optimizations if not present
    if ! grep -q "MODERN_GPU_OPTIMIZED" "$makefile"; then
        sed -i.bak '/^all:/a\
	@echo "Building with modern GPU optimizations (Ampere/Ada/Hopper)"' "$makefile"
    fi
    
    # Remove backup files
    rm -f "$makefile.bak"
done

# Update run scripts to use CUDA 12.4
find . -name "run.sh" -type f | while read -r script; do
    echo "Updating $script..."
    
    # Only update if the script contains nvcc commands
    if grep -q "nvcc" "$script"; then
        sed -i.bak 's/-arch=sm_70/-arch=sm_80/g' "$script"
        sed -i.bak 's/-arch=sm_75/-arch=sm_80/g' "$script"
        sed -i.bak 's/-arch=sm_86/-arch=sm_80/g' "$script"
        sed -i.bak 's/-arch=sm_90/-arch=sm_80/g' "$script"
        sed -i.bak 's/nvcc.*-std=c++17/nvcc -std=c++17 -DCUDA_VERSION=12.4/g' "$script"
        rm -f "$script.bak"
    fi
done

echo "CUDA version updates completed!"
echo ""
echo "Key changes made:"
echo "- Updated architecture to sm_80 for modern GPU support (Ampere/Ada/Hopper)"
echo "- Added CUDA_VERSION=12.4 definitions"
echo "- Updated nvcc flags to include CUDA version"
echo "- Added modern GPU optimization markers"
echo ""
echo "Supported architectures:"
echo "- sm_80: Ampere (A100, RTX 3000 series)"
echo "- sm_86: Ada Lovelace (RTX 4000 series)"
echo "- sm_90: Hopper (H100, H200 series)"
echo ""
echo "Please rebuild all projects with:"
echo "make clean && make"
