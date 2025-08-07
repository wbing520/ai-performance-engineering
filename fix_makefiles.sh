#!/bin/bash

# Fix script for Makefiles with proper formatting and CUDA 12.9
echo "Fixing Makefiles with proper formatting and CUDA 12.9..."

# Find all Makefiles and fix them
find . -name "Makefile" -type f | while read -r makefile; do
    echo "Fixing $makefile..."
    
    # Create a backup
    cp "$makefile" "$makefile.backup"
    
    # Fix the formatting issues and update CUDA version
    cat > "$makefile" << 'EOF'
TARGET = $(basename $(notdir $(CURDIR)))
NVCC   = nvcc
ARCH   = -arch=sm_100
OPT    = -O3
CUDA_VERSION = 12.9

all: $(TARGET)
	@echo "Building with Blackwell B200/B300 optimizations"
$(TARGET): $(TARGET).cu
	$(NVCC) $(OPT) $(ARCH) -std=c++17 -DCUDA_VERSION=$(CUDA_VERSION) -o $@ $<

clean:
	rm -f $(TARGET)
EOF
    
    # If it's a multi-target Makefile, fix it properly
    if grep -q "TARGETS" "$makefile.backup"; then
        targets=$(grep "TARGETS" "$makefile.backup" | sed 's/TARGETS = //')
        first_target=$(echo $targets | awk '{print $1}')
        
        cat > "$makefile" << EOF
TARGETS = $targets
NVCC   = nvcc
ARCH   = -arch=sm_100
OPT    = -O3
CUDA_VERSION = 12.9

all: \$(TARGETS)
	@echo "Building with Blackwell B200/B300 optimizations"

$first_target: $first_target.cu
	\$(NVCC) \$(OPT) \$(ARCH) -std=c++17 -DCUDA_VERSION=\$(CUDA_VERSION) -o \$@ \$<

clean:
	rm -f \$(TARGETS)
EOF
    fi
    
    # Remove backup
    rm -f "$makefile.backup"
done

echo "Makefiles fixed!"
