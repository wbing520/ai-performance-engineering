#!/bin/bash

# Update script for CUDA 12.9 and Blackwell B200/B300 compliance
# This script updates all Makefiles to use the latest CUDA version and optimizations

echo "Updating CUDA versions and Blackwell B200/B300 optimizations..."

# Find all Makefiles and update them
find . -name "Makefile" -type f | while read -r makefile; do
    echo "Updating $makefile..."
    
    # Update CUDA architecture to sm_100 for Blackwell B200/B300
    sed -i.bak 's/-arch=sm_70/-arch=sm_100/g' "$makefile"
    sed -i.bak 's/-arch=sm_75/-arch=sm_100/g' "$makefile"
    sed -i.bak 's/-arch=sm_80/-arch=sm_100/g' "$makefile"
    sed -i.bak 's/-arch=sm_86/-arch=sm_100/g' "$makefile"
    sed -i.bak 's/-arch=sm_89/-arch=sm_100/g' "$makefile"
    sed -i.bak 's/-arch=sm_90/-arch=sm_100/g' "$makefile"
    
    # Add CUDA version definition if not present
    if ! grep -q "CUDA_VERSION" "$makefile"; then
        # Add after the first line
        sed -i.bak '1a\
CUDA_VERSION = 12.9' "$makefile"
    fi
    
    # Update nvcc flags to include CUDA version and NVTX
    sed -i.bak 's/nvcc.*-std=c++17/nvcc -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION) -lnvtx3/g' "$makefile"
    
    # Add Blackwell B200/B300 optimizations if not present
    if ! grep -q "BLACKWELL_OPTIMIZED" "$makefile"; then
        sed -i.bak '/^all:/a\
	@echo "Building with Blackwell B200/B300 optimizations (SM100)"' "$makefile"
    fi
    
    # Add enhanced profiling targets if not present
    if ! grep -q "profile-hta" "$makefile"; then
        cat >> "$makefile" << 'EOF'

# HTA (Holistic Tracing Analysis) profiling
profile-hta: $(TARGET)
	@echo "HTA profiling for multi-GPU analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile ./$(TARGET)

# Perf profiling for system-level analysis
profile-perf: $(TARGET)
	@echo "Perf profiling for system-level analysis..."
	perf record -g -p $$(pgrep $(TARGET)) -o perf.data ./$(TARGET)
	perf report -i perf.data

# Enhanced profiling with all tools
profile-all: $(TARGET)
	@echo "Comprehensive profiling with all tools..."
	@echo "1. Nsight Systems timeline..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt -o comprehensive_timeline ./$(TARGET)
	@echo "2. Nsight Compute kernel analysis..."
	ncu --metrics achieved_occupancy,warp_execution_efficiency,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram_read_throughput,dram_write_throughput -o comprehensive_kernel ./$(TARGET)
	@echo "3. Memory profiling..."
	nsys profile --force-overwrite=true -t cuda,cudamemcpy -o comprehensive_memory ./$(TARGET)
	@echo "4. HTA analysis..."
	nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl -o comprehensive_hta ./$(TARGET)
EOF
    fi
    
    # Remove backup files
    rm -f "$makefile.bak"
done

# Update run scripts to use CUDA 12.9
find . -name "run.sh" -type f | while read -r script; do
    echo "Updating $script..."
    
    # Only update if the script contains nvcc commands
    if grep -q "nvcc" "$script"; then
        sed -i.bak 's/-arch=sm_70/-arch=sm_100/g' "$script"
        sed -i.bak 's/-arch=sm_75/-arch=sm_100/g' "$script"
        sed -i.bak 's/-arch=sm_80/-arch=sm_100/g' "$script"
        sed -i.bak 's/-arch=sm_86/-arch=sm_100/g' "$script"
        sed -i.bak 's/-arch=sm_89/-arch=sm_100/g' "$script"
        sed -i.bak 's/-arch=sm_90/-arch=sm_100/g' "$script"
        sed -i.bak 's/nvcc.*-std=c++17/nvcc -std=c++17 -DCUDA_VERSION=12.9 -lnvtx3/g' "$script"
        rm -f "$script.bak"
    fi
done

# Update Python requirements files
find . -name "requirements*.txt" -type f | while read -r req_file; do
    echo "Updating $req_file..."
    
    # Update PyTorch to latest nightly with CUDA 12.9
    sed -i.bak 's/torch==2\.[0-9]\+\.[0-9]\+/torch==2.8.0+cu129/g' "$req_file"
    sed -i.bak 's/torchvision==[0-9]\+\.[0-9]\+\.[0-9]\+/torchvision==0.23.0+cu129/g' "$req_file"
    sed -i.bak 's/torchaudio==[0-9]\+\.[0-9]\+\.[0-9]\+/torchaudio==2.8.0+cu129/g' "$req_file"
    
    # Update Triton to latest version
    sed -i.bak 's/triton==[0-9]\+\.[0-9]\+\.[0-9]\+/triton==3.4.0/g' "$req_file"
    
    # Add CUDA 12.9 runtime libraries if not present
    if ! grep -q "nvidia-cuda-runtime-cu12" "$req_file"; then
        cat >> "$req_file" << 'EOF'

# CUDA 12.9 runtime libraries
nvidia-cuda-runtime-cu12==12.9.140
nvidia-cudnn-cu12==9.0.0.29
nvidia-cublas-cu12==12.9.2.65
nvidia-cufft-cu12==11.2.2.12
nvidia-curand-cu12==10.4.0.141
nvidia-cusolver-cu12==12.2.0.141
nvidia-cusparse-cu12==12.3.0.141
nvidia-nccl-cu12==2.20.5
nvidia-nvtx-cu12==12.9.140
EOF
    fi
    
    rm -f "$req_file.bak"
done

echo "CUDA version updates completed!"
echo ""
echo "Key changes made:"
echo "- Updated architecture to sm_100 for Blackwell B200/B300 (Compute Capability 10.0)"
echo "- Added CUDA_VERSION=12.9 definitions"
echo "- Updated nvcc flags to include CUDA version and NVTX"
echo "- Added Blackwell B200/B300 optimization markers"
echo "- Added enhanced profiling targets (HTA, Perf, comprehensive profiling)"
echo "- Updated PyTorch to 2.8.0+cu129 (nightly builds)"
echo "- Updated Triton to 3.4.0"
echo "- Added CUDA 12.9 runtime libraries"
echo ""
echo "Blackwell B200/B300 Features:"
echo "- SM100 Architecture (Compute Capability 10.0)"
echo "- HBM3e Memory (up to 3.2TB/s bandwidth)"
echo "- 4th Gen Tensor Cores"
echo "- TMA (Tensor Memory Accelerator)"
echo "- Multi-GPU NVLink-C2C"
echo ""
echo "Enhanced Profiling Tools:"
echo "- Nsight Systems (latest timeline analysis)"
echo "- Nsight Compute (latest kernel analysis)"
echo "- HTA (Holistic Tracing Analysis for multi-GPU)"
echo "- Perf (system-level analysis)"
echo "- Comprehensive profiling with all tools"
echo ""
echo "Please rebuild all projects with:"
echo "make clean && make"
