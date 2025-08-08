#!/bin/bash

# Fix script for Makefiles with proper formatting and CUDA 12.4
echo "Fixing Makefiles with proper formatting and CUDA 12.4..."

# Find all Makefiles and fix them
find . -name "Makefile" -type f | while read -r makefile; do
    echo "Fixing $makefile..."
    
    # Create a backup
    cp "$makefile" "$makefile.backup"
    
    # Fix the formatting issues and update CUDA version
    cat > "$makefile" << 'EOF'
TARGET = $(basename $(notdir $(CURDIR)))
NVCC   = nvcc
ARCH   = -arch=sm_80
OPT    = -O3
CUDA_VERSION = 12.4

all: $(TARGET)
	@echo "Building with modern GPU optimizations (Ampere/Ada/Hopper)"
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
ARCH   = -arch=sm_80
OPT    = -O3
CUDA_VERSION = 12.4

all: \$(TARGETS)
	@echo "Building with modern GPU optimizations (Ampere/Ada/Hopper)"

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
