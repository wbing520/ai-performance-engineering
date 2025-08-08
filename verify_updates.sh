#!/bin/bash

# Verification script for CUDA 12.4 and modern GPU compliance
echo "Verifying CUDA 12.4 and modern GPU compliance..."

# Check Makefiles for proper updates
echo "Checking Makefiles..."
makefile_count=0
updated_count=0

find . -name "Makefile" -type f | while read -r makefile; do
    makefile_count=$((makefile_count + 1))
    
    # Check for CUDA_VERSION definition
    if grep -q "CUDA_VERSION = 12.4" "$makefile"; then
        updated_count=$((updated_count + 1))
        echo "✅ $makefile - CUDA_VERSION correctly set to 12.4"
    else
        echo "❌ $makefile - Missing or incorrect CUDA_VERSION"
    fi
    
    # Check for modern architecture support
    if grep -q "-arch=sm_80\|-arch=sm_86\|-arch=sm_90" "$makefile"; then
        echo "✅ $makefile - Architecture supports modern GPUs"
    else
        echo "❌ $makefile - Architecture not updated to modern GPUs"
    fi
    
    # Check for modern GPU optimizations
    if grep -q "Building with modern GPU optimizations" "$makefile"; then
        echo "✅ $makefile - Modern GPU optimizations added"
    else
        echo "❌ $makefile - Missing modern GPU optimizations"
    fi
done

echo ""
echo "Summary:"
echo "- Total Makefiles found: $makefile_count"
echo "- Updated with CUDA_VERSION: $updated_count"

# Check for PyTorch 2.8 features
echo ""
echo "Checking PyTorch files for 2.8 features..."
if grep -r "torch.compile.*mode.*max-autotune" code/; then
    echo "✅ torch.compile with max-autotune found"
else
    echo "❌ torch.compile with max-autotune not found"
fi

if grep -r "torch._dynamo.config" code/; then
    echo "✅ TorchDynamo configuration found"
else
    echo "❌ TorchDynamo configuration not found"
fi

# Check for CUDA 12.4 features
echo ""
echo "Checking for CUDA 12.4 features..."
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

# Check for modern GPU support
echo ""
echo "Checking for modern GPU support..."
if grep -r "sm_80\|sm_86\|sm_90" code/; then
    echo "✅ Modern GPU architecture support found"
else
    echo "❌ Modern GPU architecture support not found"
fi

echo ""
echo "Verification complete!"
