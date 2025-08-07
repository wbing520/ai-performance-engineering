#!/bin/bash

# Update script for CUDA 12.9 and Blackwell B200/B300 compliance
# This script updates all Makefiles to use the latest CUDA version and optimizations

echo "Updating CUDA versions and Blackwell optimizations..."

# Find all Makefiles and update them
find . -name "Makefile" -type f | while read -r makefile; do
    echo "Updating $makefile..."
    
    # Update CUDA architecture to sm_100 for Blackwell B200/B300
    sed -i.bak 's/-arch=sm_90/-arch=sm_100/g' "$makefile"
    sed -i.bak 's/-arch=sm_90a/-arch=sm_100/g' "$makefile"
    
    # Add CUDA version definition if not present
    if ! grep -q "CUDA_VERSION" "$makefile"; then
        # Add after the first line
        sed -i.bak '1a\
CUDA_VERSION = 12.9' "$makefile"
    fi
    
    # Update nvcc flags to include CUDA version
    sed -i.bak 's/nvcc.*-std=c++17/nvcc -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION)/g' "$makefile"
    
    # Add Blackwell optimizations if not present
    if ! grep -q "BLACKWELL_OPTIMIZED" "$makefile"; then
        sed -i.bak '/^all:/a\
	@echo "Building with Blackwell B200/B300 optimizations"' "$makefile"
    fi
    
    # Remove backup files
    rm -f "$makefile.bak"
done

# Update run scripts to use CUDA 12.9
find . -name "run.sh" -type f | while read -r script; do
    echo "Updating $script..."
    
    # Only update if the script contains nvcc commands
    if grep -q "nvcc" "$script"; then
        sed -i.bak 's/-arch=sm_90/-arch=sm_100/g' "$script"
        sed -i.bak 's/-arch=sm_90a/-arch=sm_100/g' "$script"
        sed -i.bak 's/nvcc.*-std=c++17/nvcc -std=c++17 -DCUDA_VERSION=12.9/g' "$script"
        rm -f "$script.bak"
    fi
done

echo "CUDA version updates completed!"
echo ""
echo "Key changes made:"
echo "- Updated architecture from sm_90/sm_90a to sm_100 for Blackwell B200/B300"
echo "- Added CUDA_VERSION=12.9 definitions"
echo "- Updated nvcc flags to include CUDA version"
echo "- Added Blackwell optimization markers"
echo ""
echo "Please rebuild all projects with:"
echo "make clean && make"
