#!/bin/bash

# Verification script for CUDA 12.9 and Blackwell B200/B300 compliance
echo "Verifying CUDA 12.9 and Blackwell B200/B300 compliance..."

# Check Makefiles for proper updates
echo "Checking Makefiles..."
makefile_count=0
updated_count=0

find . -name "Makefile" -type f | while read -r makefile; do
    makefile_count=$((makefile_count + 1))
    
    # Check for CUDA_VERSION definition
    if grep -q "CUDA_VERSION = 12.9" "$makefile"; then
        updated_count=$((updated_count + 1))
        echo "✅ $makefile - CUDA_VERSION correctly set to 12.9"
    else
        echo "❌ $makefile - Missing or incorrect CUDA_VERSION"
    fi
    
    # Check for sm_100 architecture
    if grep -q "-arch=sm_100" "$makefile"; then
        echo "✅ $makefile - Architecture correctly set to sm_100"
    else
        echo "❌ $makefile - Architecture not updated to sm_100"
    fi
    
    # Check for Blackwell optimizations
    if grep -q "Building with Blackwell B200/B300 optimizations" "$makefile"; then
        echo "✅ $makefile - Blackwell optimizations added"
    else
        echo "❌ $makefile - Missing Blackwell optimizations"
    fi
done

echo ""
echo "Summary:"
echo "- Total Makefiles found: $makefile_count"
echo "- Updated with CUDA_VERSION: $updated_count"

# Check for PyTorch 2.8 nightly features
echo ""
echo "Checking PyTorch files for 2.8 nightly features..."
if grep -r "torch.compile.*mode.*max-autotune" code/; then
    echo "✅ torch.compile with max-autotune found"
else
    echo "❌ torch.compile with max-autotune not found"
fi

if grep -r "torch._dynamo.config.automatic_dynamic_shapes" code/; then
    echo "✅ TorchDynamo automatic dynamic shapes found"
else
    echo "❌ TorchDynamo automatic dynamic shapes not found"
fi

# Check for CUDA 12.9 features
echo ""
echo "Checking for CUDA 12.9 features..."
if grep -r "cudaMallocAsync" code/; then
    echo "✅ Stream-ordered memory allocation found"
else
    echo "❌ Stream-ordered memory allocation not found"
fi

if grep -r "cudaFreeAsync" code/; then
    echo "✅ Stream-ordered memory deallocation found"
else
    echo "❌ Stream-ordered memory deallocation not found"
fi

# Check for Triton 3.4 features
echo ""
echo "Checking for Triton 3.4 features..."
if grep -r "triton>=3.4" requirements_latest.txt; then
    echo "✅ Triton 3.4 requirement found"
else
    echo "❌ Triton 3.4 requirement not found"
fi

echo ""
echo "Verification complete!"
