#!/bin/bash

# Comprehensive Build Script for Architecture Switching
# Supports Hopper H100/H200 and Blackwell B200/B300
# Updated for PyTorch 2.8, CUDA 12.8, and Triton 3.3

set -e

echo "=== AI Performance Engineering - Comprehensive Build ==="
echo "PyTorch 2.8, CUDA 12.8, Triton 3.3, Architecture Switching Support"
echo ""

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

# Function to check CUDA version
check_cuda_version() {
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        echo "CUDA Version: $cuda_version"
        
        # Check if CUDA 12.8 is available
        if [[ "$cuda_version" == "12.8"* ]]; then
            echo "✓ CUDA 12.8 detected"
            return 0
        else
            echo "⚠ CUDA 12.8 not detected, using available version: $cuda_version"
            return 1
        fi
    else
        echo "⚠ nvcc not found, CUDA compilation may not work"
        return 1
    fi
}

# Function to check PyTorch version
check_pytorch_version() {
    if command -v python &> /dev/null; then
        pytorch_version=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
        echo "PyTorch Version: $pytorch_version"
        
        if [[ "$pytorch_version" == "2.8"* ]]; then
            echo "✓ PyTorch 2.8 detected"
            return 0
        else
            echo "⚠ PyTorch 2.8 not detected, using available version: $pytorch_version"
            return 1
        fi
    else
        echo "⚠ Python not found"
        return 1
    fi
}

# Function to check Triton version
check_triton_version() {
    if command -v python &> /dev/null; then
        triton_version=$(python -c "import triton; print(triton.__version__)" 2>/dev/null || echo "Not installed")
        echo "Triton Version: $triton_version"
        
        if [[ "$triton_version" == "3.3"* ]]; then
            echo "✓ Triton 3.3 detected"
            return 0
        else
            echo "⚠ Triton 3.3 not detected, using available version: $triton_version"
            return 1
        fi
    else
        echo "⚠ Python not found"
        return 1
    fi
}

# Function to check profiling tools
check_profiling_tools() {
    echo "Checking profiling tools..."
    
    # Check Nsight Systems
    if command -v nsys &> /dev/null; then
        nsys_version=$(nsys --version 2>/dev/null | head -1 || echo "Unknown")
        echo "✓ Nsight Systems: $nsys_version"
    else
        echo "⚠ Nsight Systems not found"
    fi
    
    # Check Nsight Compute
    if command -v ncu &> /dev/null; then
        ncu_version=$(ncu --version 2>/dev/null | head -1 || echo "Unknown")
        echo "✓ Nsight Compute: $ncu_version"
    else
        echo "⚠ Nsight Compute not found"
    fi
    
    # Check Perf
    if command -v perf &> /dev/null; then
        echo "✓ Perf available"
    else
        echo "⚠ Perf not found"
    fi
}

# Detect current architecture
CURRENT_ARCH=$(detect_architecture)
echo "Detected architecture: $CURRENT_ARCH"

# Check versions
check_cuda_version
check_pytorch_version
check_triton_version
check_profiling_tools

echo ""

# Install dependencies (skip torch packages if already installed)
echo "Installing dependencies..."
TMP_REQ="/tmp/ai_perf_req_$(date +%s).txt"
if python - << 'PY'
import sys
try:
    import torch
    print("torch_present")
except Exception:
    pass
PY
then
  echo "✓ torch already installed; filtering torch packages from requirements"
  sed '/^torch\(vision\|audio\)\?==/d;/^--index-url/d' requirements_latest.txt > "$TMP_REQ"
  pip install -r "$TMP_REQ"
  rm -f "$TMP_REQ"
else
  pip install -r requirements_latest.txt
fi

# Test architecture configuration
echo "Testing architecture configuration..."
python -c "
import sys
sys.path.append('code')
from arch_config import arch_config, configure_optimizations
arch_config.print_info()
configure_optimizations()
print('✓ Architecture configuration successful')
"

# Build all CUDA projects with enhanced features
echo "Building CUDA projects with latest features..."
# Build only active chapter examples; skip archived directories
find . -name "Makefile" -type f ! -path "./archive/*" | while read -r makefile; do
    dir=$(dirname "$makefile")
    echo "Building $dir with $CURRENT_ARCH..."
    cd "$dir"
    
    # Clean previous builds
    make clean 2>/dev/null || true
    
    # Build with current architecture
    make ARCH=$CURRENT_ARCH
    
    # Test the build if executable exists
    target_name=$(basename "$dir")
    if [ -f "$target_name" ]; then
        echo "✓ Built $target_name successfully"
    fi
    
    cd - > /dev/null
done

# Build Python examples with latest features
echo "Building Python examples with PyTorch 2.8 optimizations..."
find code -name "*.py" -path "*/ch*/*" | while read -r pyfile; do
    dir=$(dirname "$pyfile")
    filename=$(basename "$pyfile")
    
    # Skip test files and utilities
    if [[ "$filename" == test_* ]] || [[ "$filename" == *_test.py ]] || [[ "$filename" == "arch_config.py" ]]; then
        continue
    fi
    
    echo "Testing $pyfile..."
    cd "$dir"
    
    # Test Python file with syntax check
    if python -m py_compile "$filename" 2>/dev/null; then
        echo "✓ $filename syntax OK"
    else
        echo "⚠ $filename syntax issues"
    fi
    
    cd - > /dev/null
done

# Run comprehensive tests
echo "Running comprehensive tests..."
python test_architecture_switching.py

# Test profiling tools
echo "Testing profiling tools..."
if command -v nsys &> /dev/null; then
    echo "✓ Nsight Systems available"
fi

if command -v ncu &> /dev/null; then
    echo "✓ Nsight Compute available"
fi

if command -v perf &> /dev/null; then
    echo "✓ Perf available"
fi

# Generate build report
echo "Generating build report..."
cat > "build_report_$(date +%Y%m%d_%H%M%S).md" << EOF
# AI Performance Engineering Build Report

## Build Information
- **Timestamp**: $(date)
- **Architecture**: $CURRENT_ARCH
- **CUDA Version**: $(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' || echo "Unknown")
- **PyTorch Version**: $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Unknown")
- **Triton Version**: $(python -c "import triton; print(triton.__version__)" 2>/dev/null || echo "Unknown")

## Architecture Details
EOF

if [ "$CURRENT_ARCH" = "sm_90" ]; then
    cat >> "build_report_$(date +%Y%m%d_%H%M%S).md" << EOF
- **GPU**: Hopper H100/H200
- **Compute Capability**: 9.0
- **Memory**: HBM3
- **Features**: Transformer Engine, Dynamic Programming, TMA
EOF
elif [ "$CURRENT_ARCH" = "sm_100" ]; then
    cat >> "build_report_$(date +%Y%m%d_%H%M%S).md" << EOF
- **GPU**: Blackwell B200/B300
- **Compute Capability**: 10.0
- **Memory**: HBM3e
- **Features**: TMA, NVLink-C2C, Stream-ordered Memory
EOF
fi

cat >> "build_report_$(date +%Y%m%d_%H%M%S).md" << EOF

## Build Status
- **CUDA Projects**: Built successfully
- **Python Examples**: Syntax checked
- **Architecture Configuration**: Tested
- **Profiling Tools**: Available

## Next Steps
1. Run examples: \`python code/ch1/performance_basics.py\`
2. Profile performance: \`bash code/profiler_scripts/comprehensive_profile.sh script.py\`
3. Test architecture switching: \`bash code/switch_architecture.sh sm_100\`
4. View build report: \`cat build_report_$(date +%Y%m%d_%H%M%S).md\`
EOF

echo "Build completed successfully!"
echo "Build report generated: build_report_$(date +%Y%m%d_%H%M%S).md"
echo ""
echo "To run examples:"
echo "  python code/ch1/performance_basics.py"
echo "  python code/ch2/hardware_info.py"
echo ""
echo "To profile performance:"
echo "  bash code/profiler_scripts/comprehensive_profile.sh script.py"
echo "  bash code/profiler_scripts/pytorch_profile.sh script.py"
echo ""
echo "To switch architectures:"
echo "  bash code/switch_architecture.sh sm_90  # Hopper H100/H200"
echo "  bash code/switch_architecture.sh sm_100 # Blackwell B200/B300"
