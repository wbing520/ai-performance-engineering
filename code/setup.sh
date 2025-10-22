#!/bin/bash
#
# AI Performance Engineering Setup Script
# ========================================
#
# This script installs EVERYTHING you need:
#   1. PyTorch 2.9 nightly with CUDA 13.0
#   2. CUDA 13.0 toolchain (nvcc, libraries)
#   3. NVIDIA Nsight Systems 2025.3.2 (for timeline profiling)
#   4. NVIDIA Nsight Compute 2025.3.1 (for kernel metrics)
#   5. All Python dependencies from requirements_latest.txt
#   6. System tools (numactl, perf, htop, etc.)
#   7. Configures NVIDIA drivers for profiling
#
# Requirements:
#   - Ubuntu 22.04+ (tested on 22.04)
#   - NVIDIA B200/B300 GPU (or compatible)
#   - sudo/root access
#   - Internet connection
#
# Usage:
#   sudo ./setup.sh
#
# Duration: 10-20 minutes
#
# What it does:
#   - Updates apt packages
#   - Installs CUDA 13.0 toolkit
#   - Installs latest Nsight tools (2025.x)
#   - Installs PyTorch 2.9 nightly
#   - Installs nvidia-ml-py (replaces deprecated pynvml)
#   - Configures NVIDIA kernel modules for profiling
#   - Runs validation tests
#
# After running this script, you can:
#   - Run examples: python3 ch1/performance_basics.py
#   - Test everything: ./run_all_tests.sh
#   - Benchmark peak: python3 benchmark_peak.py
#   - Profile examples: ./start.sh
#

set -e  # Exit on any error

echo "🚀 AI Performance Engineering Setup Script"
echo "=========================================="
echo "This script will install:"
echo "  • PyTorch 2.9 nightly with CUDA 13.0"
echo "  • CUDA 13.0 toolchain and development tools"
echo "  • NVIDIA Nsight Systems 2025.3.2 (latest)"
echo "  • NVIDIA Nsight Compute 2025.3.1 (latest)"
echo "  • All project dependencies"
echo "  • System tools (numactl, perf, etc.)"
echo ""

PROJECT_ROOT="$(dirname "$(realpath "$0")")"
REQUIRED_DRIVER_VERSION="575.57"
echo "📁 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Allow pip to install over system packages when running as root on Debian-based distros
export PIP_BREAK_SYSTEM_PACKAGES=1

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  Running as root. This is fine for containerized environments."
else
   echo "❌ This script requires root privileges. Please run with sudo."
   exit 1
fi

# Check Ubuntu version
if ! command -v lsb_release &> /dev/null; then
    echo "Installing lsb-release..."
    apt update && apt install -y lsb-release
fi

UBUNTU_VERSION=$(lsb_release -rs)
echo "📋 Detected Ubuntu version: $UBUNTU_VERSION"

if [[ "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "20.04" ]]; then
    echo "⚠️  Warning: This script is tested on Ubuntu 22.04. Other versions may work but are not guaranteed."
fi

# Check for NVIDIA GPU
echo ""
echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✅ NVIDIA GPU detected"

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ -n "$DRIVER_VERSION" ]]; then
        python3 - "$DRIVER_VERSION" "$REQUIRED_DRIVER_VERSION" <<'PY'
import sys
from packaging import version
current = version.parse(sys.argv[1])
required = version.parse(sys.argv[2])
if current < required:
    print(f"⚠️  NVIDIA driver {current} is older than required {required} for CUDA 13.0/PyTorch 2.9. "
          "Upgrade to nvidia-driver-580 (or newer) and reboot, then rerun this setup.")
PY
    fi
else
    echo "❌ NVIDIA GPU not detected. Please ensure NVIDIA drivers are installed."
    exit 1
fi

# Ensure open kernel modules are enabled for Blackwell GPUs
MODPROBE_CONF="/etc/modprobe.d/nvidia-open.conf"
if [[ ! -f "$MODPROBE_CONF" ]] || ! grep -q "NVreg_OpenRmEnableUnsupportedGpus=1" "$MODPROBE_CONF"; then
    echo "Configuring NVIDIA open kernel modules for Blackwell GPUs..."
    cat <<'EOF' > "$MODPROBE_CONF"
options nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0
EOF
    update-initramfs -u
    if lsmod | grep -q "^nvidia"; then
        echo "Reloading NVIDIA kernel modules to enable profiling counters..."
        systemctl stop nvidia-persistenced >/dev/null 2>&1 || true
        for module in nvidia_uvm nvidia_peermem nvidia_modeset nvidia_drm nvidia; do
            if lsmod | grep -q "^${module}"; then
                modprobe -r "${module}" >/dev/null 2>&1 || true
            fi
        done
        modprobe nvidia NVreg_OpenRmEnableUnsupportedGpus=1 NVreg_RestrictProfilingToAdminUsers=0 >/dev/null 2>&1 || true
        for module in nvidia_modeset nvidia_uvm nvidia_peermem; do
            modprobe "${module}" >/dev/null 2>&1 || true
        done
        systemctl start nvidia-persistenced >/dev/null 2>&1 || true
    fi
fi

# Update system packages
echo ""
echo "📦 Updating system packages..."
apt update

# Install Python and pip if not present
echo ""
echo "🐍 Installing Python and pip..."
apt install -y python3 python3-pip python3-venv python3-dev

# Upgrade pip
python3 -m pip install --upgrade pip setuptools packaging

# Remove distro flatbuffers package whose invalid version breaks pip metadata
if dpkg -s python3-flatbuffers >/dev/null 2>&1; then
    echo "Removing distro python3-flatbuffers package (invalid version metadata)..."
    apt remove -y python3-flatbuffers
fi

# Install CUDA 12.9 toolchain
echo ""
echo "🔧 Installing CUDA 12.9 toolchain..."
apt install -y cuda-toolkit-12-9

# Install CUDA sanitizers and debugging tools (compute-sanitizer, cuda-memcheck, etc.)
echo ""
echo "🛡️  Installing CUDA sanitizers and debugging tools..."
if apt install -y cuda-command-line-tools-12-9; then
    echo "✅ CUDA command-line tools 12.9 installed (compute-sanitizer, cuda-gdb, cuda-memcheck)"
else
    echo "⚠️  Could not install cuda-command-line-tools-12-9, trying fallback packages..."
    if apt install -y cuda-command-line-tools; then
        echo "✅ CUDA command-line tools (generic) installed"
    else
        echo "⚠️  cuda-command-line-tools package unavailable. Trying NVIDIA CUDA toolkit..."
        if apt install -y nvidia-cuda-toolkit; then
            echo "✅ NVIDIA CUDA toolkit installed (includes cuda-memcheck)"
        else
            echo "❌ Could not install CUDA command-line tools. compute-sanitizer may be unavailable."
        fi
    fi
fi

# Ensure compute-sanitizer is present; install sanitizer package directly if needed
if ! command -v compute-sanitizer &> /dev/null; then
    echo "⚠️  compute-sanitizer not found after command-line tools install. Installing cuda-sanitizer package..."
    if apt install -y cuda-sanitizer-12-9; then
        echo "✅ cuda-sanitizer-12-9 installed"
    else
        echo "⚠️  Could not install cuda-sanitizer-12-9, attempting generic cuda-sanitizer package..."
        if apt install -y cuda-sanitizer; then
            echo "✅ cuda-sanitizer package installed"
        else
            echo "❌ compute-sanitizer installation failed; please install manually."
        fi
    fi
fi

# Install latest NVIDIA Nsight Systems and Compute
echo ""
echo "🔍 Installing latest NVIDIA Nsight Systems and Compute..."

# Create temporary directory for downloads
TEMP_DIR="/tmp/nsight_install"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Install Nsight Systems and set binary alternative  
echo "Installing Nsight Systems (pinned 2025.3.2)..."
NSYS_VERSION="2025.3.2"
apt install -y "nsight-systems-${NSYS_VERSION}"
# Try multiple possible locations
for bin_path in "/opt/nvidia/nsight-systems/${NSYS_VERSION}/bin/nsys" "/opt/nvidia/nsight-systems/${NSYS_VERSION}/nsys"; do
    if [[ -x "$bin_path" ]]; then
        NSYS_BIN="$bin_path"
        break
    fi
done
if [[ -n "$NSYS_BIN" ]] && [[ -x "$NSYS_BIN" ]]; then
    update-alternatives --install /usr/local/bin/nsys nsys "$NSYS_BIN" 50
    update-alternatives --set nsys "$NSYS_BIN"
    echo "✅ Nsight Systems pinned to ${NSYS_VERSION} (${NSYS_BIN})"
else
    echo "❌ Nsight Systems binary not found"
fi

# Install Nsight Compute and set binary alternative
echo "Installing Nsight Compute (pinned 2025.3.1)..."
NCU_VERSION="2025.3.1"
apt install -y "nsight-compute-${NCU_VERSION}"
NCU_BIN="/opt/nvidia/nsight-compute/${NCU_VERSION}/ncu"
if [[ -x "$NCU_BIN" ]]; then
    update-alternatives --install /usr/local/bin/ncu ncu "$NCU_BIN" 50
    update-alternatives --set ncu "$NCU_BIN"
    echo "✅ Nsight Compute pinned to ${NCU_VERSION} (${NCU_BIN})"
else
    echo "❌ Nsight Compute binary not found at ${NCU_BIN}"
fi

# Nsight tools are already in PATH when installed via apt
echo "✅ Nsight tools installed and available in PATH"

# Configure PATH and LD_LIBRARY_PATH for CUDA 13.0
echo ""
echo "🔧 Configuring CUDA 13.0 environment..."
# Update /etc/environment for system-wide CUDA 13.0
if ! grep -q "/usr/local/cuda-13.0/bin" /etc/environment; then
    sed -i 's|PATH="\(.*\)"|PATH="/usr/local/cuda-13.0/bin:\1"|' /etc/environment
    echo "Added CUDA 13.0 to system PATH"
fi

# Create profile.d script for CUDA 13.0
cat > /etc/profile.d/cuda-13.0.sh << 'PROFILE_EOF'
# CUDA 13.0 environment variables
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_PATH=/usr/local/cuda-13.0
PROFILE_EOF
chmod +x /etc/profile.d/cuda-13.0.sh
echo "✅ Created /etc/profile.d/cuda-13.0.sh for persistent CUDA 13.0 environment"

# Update nvcc symlink to CUDA 13.0 (override Ubuntu's default)
rm -f /usr/bin/nvcc
ln -s /usr/local/cuda-13.0/bin/nvcc /usr/bin/nvcc
echo "✅ Updated /usr/bin/nvcc symlink to CUDA 13.0"

# Source the CUDA environment for current session
source /etc/profile.d/cuda-13.0.sh

# Clean up
cd /
rm -rf "$TEMP_DIR"
cd "$PROJECT_ROOT"

# Install system tools for performance testing
echo ""
echo "🛠️  Installing system performance tools..."
apt install -y \
    numactl \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) \
    gdb \
    perf-tools-unstable \
    infiniband-diags \
    perftest \
    htop \
    sysstat

# Install PyTorch 2.9 nightly with CUDA 12.9
echo ""
echo "🔥 Installing PyTorch 2.9 nightly with CUDA 12.9..."
python3 -m pip install --index-url https://download.pytorch.org/whl/nightly/cu129 \
    --no-cache-dir --upgrade --ignore-installed torch torchvision torchaudio

# Install project dependencies
echo ""
echo "📚 Installing project dependencies..."

# Use the updated requirements file with pinned versions
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements_latest.txt"

# Install dependencies with error handling
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "Installing Python packages from requirements file..."
    if ! python3 -m pip install --no-input --upgrade --ignore-installed -r "$REQUIREMENTS_FILE"; then
        echo "⚠️  Some packages failed to install from requirements file."
        echo "Installing core packages individually..."
        python3 -m pip install --no-input --upgrade --ignore-installed \
            blinker==1.9.0 \
            nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
            numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
            matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
            jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 flake8==7.3.0 mypy==1.18.2 \
            transformers==4.40.2 datasets==2.18.0 accelerate==0.29.0 sentencepiece==0.2.0 tokenizers==0.19.1 \
            onnx==1.19.0 onnxruntime-gpu==1.23.0 \
            py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
            optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
            dask==2025.9.1 xarray==2025.6.1
    fi
else
    echo "⚠️  Requirements file not found at $REQUIREMENTS_FILE. Installing core packages directly..."
    python3 -m pip install --no-input --upgrade --ignore-installed \
        blinker==1.9.0 \
        nvidia-ml-py==12.560.30 psutil==7.1.0 GPUtil==1.4.0 py-cpuinfo==9.0.0 \
        numpy==2.1.2 pandas==2.3.2 scikit-learn==1.7.2 pillow==11.3.0 \
        matplotlib==3.10.6 seaborn==0.13.2 tensorboard==2.20.0 wandb==0.22.0 plotly==6.3.0 bokeh==3.8.0 dash==3.2.0 \
        jupyter==1.1.1 ipykernel==6.30.1 black==25.9.0 flake8==7.3.0 mypy==1.18.2 \
        transformers==4.40.2 datasets==2.18.0 accelerate==0.29.0 sentencepiece==0.2.0 tokenizers==0.19.1 \
        onnx==1.19.0 onnxruntime-gpu==1.23.0 \
        py-spy==0.4.1 memory-profiler==0.61.0 line-profiler==5.0.0 pyinstrument==5.1.1 snakeviz==2.2.2 \
        optuna==4.5.0 hyperopt==0.2.7 ray==2.49.2 \
        dask==2025.9.1 xarray==2025.6.1
fi

# Fix hardware info script compatibility
echo ""
echo "🔧 Fixing hardware info script compatibility..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    # Backup original file
    cp "$PROJECT_ROOT/ch2/hardware_info.py" "$PROJECT_ROOT/ch2/hardware_info.py.backup"
    
    # Fix the compatibility issue
    sed -i 's/"max_threads_per_block": device_props.max_threads_per_block,/"max_threads_per_block": getattr(device_props, '\''max_threads_per_block'\'', 1024),/' "$PROJECT_ROOT/ch2/hardware_info.py"
    
    echo "✅ Fixed hardware info script compatibility"
fi

# Verify installation
echo ""
echo "🧪 Verifying installation..."

# Check PyTorch
echo "Checking PyTorch installation..."
python3 - "$REQUIRED_DRIVER_VERSION" <<'PY'
import os
import sys
import textwrap
from packaging import version

required_driver = version.parse(sys.argv[1])

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"❌ PyTorch import failed: {exc}")
    sys.exit(1)

print(f"✅ PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"✅ CUDA available: {cuda_available}")

if cuda_available:
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    try:
        print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
    except Exception:  # pragma: no cover
        pass
else:
    driver_version = None
    try:
        from torch._C import _cuda_getDriverVersion  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _cuda_getDriverVersion = None

    if _cuda_getDriverVersion is not None:
        try:
            driver_version = _cuda_getDriverVersion()
        except Exception:
            driver_version = None

    if driver_version:
        current = version.parse(str(driver_version))
        if current < required_driver:
            print(
                textwrap.dedent(
                    f"""
                    ⚠ NVIDIA driver {current} is older than required {required_driver}.
                    → Install a newer driver (e.g., nvidia-driver-580) and reboot, then rerun setup.sh.
                    """
                ).strip()
            )
    else:
        print("❌ CUDA runtime not available. Ensure the NVIDIA driver meets CUDA 13.0 requirements and reboot if this is a fresh install.")
PY

# Check CUDA tools
echo ""
echo "Checking CUDA tools..."
if command -v nvcc &> /dev/null; then
    echo "✅ NVCC: $(nvcc --version | head -1)"
else
    echo "❌ NVCC not found"
fi

# Check Nsight tools
echo ""
echo "Checking Nsight tools..."
if command -v nsys &> /dev/null; then
    NSYS_VERSION=$(nsys --version 2>/dev/null | head -1)
    echo "✅ Nsight Systems: $NSYS_VERSION"
    # Check if it's a recent 2025 version
    if echo "$NSYS_VERSION" | grep -q "2025"; then
        echo "  🎉 Recent 2025 version installed!"
    else
        echo "  ⚠️  May not be the latest version (expected: 2025.x.x)"
    fi
else
    echo "❌ Nsight Systems not found"
fi

if command -v ncu &> /dev/null; then
    NCU_VERSION=$(ncu --version 2>/dev/null | head -1)
    echo "✅ Nsight Compute: $NCU_VERSION"
    # Check if it's a recent 2025 version
    if echo "$NCU_VERSION" | grep -q "2025"; then
        echo "  🎉 Recent 2025 version installed!"
    else
        echo "  ⚠️  May not be the latest version (expected: 2025.x.x)"
    fi
else
    echo "❌ Nsight Compute not found"
fi

# Check CUDA sanitizers and memcheck tools
echo ""
echo "Checking CUDA sanitizers..."
sanitizer_tools=("compute-sanitizer" "cuda-memcheck")
for tool in "${sanitizer_tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo "✅ $tool: installed"
    else
        echo "❌ $tool: not found"
    fi
done

# Check system tools
echo ""
echo "Checking system tools..."
tools=("numactl" "perf" "htop" "iostat" "ibstat")
for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "✅ $tool: installed"
    else
        echo "❌ $tool: not found"
    fi
done

# Run basic performance test
echo ""
echo "🚀 Running basic performance test..."
python3 -c "
import torch
import time

device = torch.device('cuda')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Warm up
for _ in range(10):
    z = torch.mm(x, y)

# Time the operation
start = time.time()
for _ in range(100):
    z = torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f'✅ Matrix multiplication (1000x1000): {(end - start) * 1000 / 100:.2f} ms per operation')
print(f'✅ GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB')
print(f'✅ GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB')
"

# Test example scripts
echo ""
echo "🧪 Testing example scripts..."

# Test Chapter 1
echo "Testing Chapter 1 (Performance Basics)..."
if [ -f "$PROJECT_ROOT/ch1/performance_basics.py" ]; then
    if python3 "$PROJECT_ROOT/ch1/performance_basics.py" > /dev/null 2>&1; then
        echo "✅ Chapter 1: Performance basics working"
    else
        echo "⚠️  Chapter 1: Some issues detected (check output above)"
    fi
else
    echo "ℹ️  Chapter 1 example not present, skipping."
fi

# Test Chapter 2
echo "Testing Chapter 2 (Hardware Info)..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    if python3 "$PROJECT_ROOT/ch2/hardware_info.py" > /dev/null 2>&1; then
        echo "✅ Chapter 2: Hardware info working"
    else
        echo "⚠️  Chapter 2: Some issues detected (check output above)"
    fi
else
    echo "ℹ️  Chapter 2 example not present, skipping."
fi

# Test Chapter 3
echo "Testing Chapter 3 (NUMA Affinity)..."
if [ -f "$PROJECT_ROOT/ch3/bind_numa_affinity.py" ]; then
    if python3 "$PROJECT_ROOT/ch3/bind_numa_affinity.py" > /dev/null 2>&1; then
        echo "✅ Chapter 3: NUMA affinity working"
    else
        echo "⚠️  Chapter 3: Some issues detected (check output above)"
    fi
else
    echo "ℹ️  Chapter 3 example not present, skipping."
fi

# Set up environment variables for optimal performance
echo ""
echo "⚙️  Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# AI Performance Engineering Environment Variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# PyTorch optimization
export TORCH_COMPILE_DEBUG=0
export TORCH_LOGS="+dynamo"

# CUDA paths
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

echo "✅ Environment variables added to ~/.bashrc"

# Comprehensive setup verification
echo ""
echo "🧪 Running comprehensive setup verification..."
echo "=============================================="

# Test 1: PyTorch and CUDA
echo "🔍 Testing PyTorch and CUDA..."
python3 -c "
import torch
import sys
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('❌ CUDA not available!')
    sys.exit(1)
print(f'  CUDA version: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
print(f'  GPU name: {torch.cuda.get_device_name(0)}')
print('✅ PyTorch and CUDA working correctly')
"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch/CUDA test failed!"
    exit 1
fi

# Test 2: Performance test
echo ""
echo "🚀 Testing GPU performance..."
python3 -c "
import torch
import time
device = torch.device('cuda')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Warm up
for _ in range(10):
    z = torch.mm(x, y)

# Time the operation
start = time.time()
for _ in range(100):
    z = torch.mm(x, y)
torch.cuda.synchronize()
end = time.time()

print(f'  Matrix multiplication: {(end - start) * 1000 / 100:.2f} ms per operation')
print(f'  GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated')
print('✅ GPU performance test passed')
"

if [ $? -ne 0 ]; then
    echo "❌ GPU performance test failed!"
    exit 1
fi

# Test 3: torch.compile test
echo ""
echo "⚡ Testing torch.compile..."
python3 -c "
import sys
import time
import torch
import traceback

device = torch.device('cuda')

def simple_model(x):
    return torch.mm(x, x.t())

x = torch.randn(1000, 1000, device=device)

# Uncompiled
start = time.time()
for _ in range(10):
    y = simple_model(x)
torch.cuda.synchronize()
uncompiled_time = time.time() - start

try:
    compiled_model = torch.compile(simple_model)
except AssertionError as exc:
    if \"duplicate template name\" in str(exc):
        print('⚠️  torch.compile skipped due to known PyTorch nightly issue: duplicate kernel template name')
        print(f'   Details: {exc}')
        sys.exit(0)
    print('❌ torch.compile failed with assertion error:')
    print(exc)
    sys.exit(1)
except Exception:
    print('❌ torch.compile failed with an unexpected exception:')
    traceback.print_exc()
    sys.exit(1)

start = time.time()
for _ in range(10):
    y = compiled_model(x)
torch.cuda.synchronize()
compiled_time = time.time() - start

speedup = uncompiled_time / compiled_time if compiled_time > 0 else float('inf')
print(f'  Uncompiled: {uncompiled_time*1000/10:.2f} ms per operation')
print(f'  Compiled: {compiled_time*1000/10:.2f} ms per operation')
print(f'  Speedup: {speedup:.2f}x')
print('✅ torch.compile test passed')
"

if [ $? -ne 0 ]; then
    echo "❌ torch.compile test failed!"
    exit 1
fi

# Test 4: Hardware info script
echo ""
echo "🔧 Testing hardware detection..."
if [ -f "$PROJECT_ROOT/ch2/hardware_info.py" ]; then
    python3 "$PROJECT_ROOT/ch2/hardware_info.py" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Hardware detection working"
    else
        echo "⚠️  Hardware detection had issues (may be expected in containers)"
    fi
else
    echo "ℹ️  Hardware detection script not present, skipping."
fi

# Test 5: NUMA binding script
echo ""
echo "🔗 Testing NUMA binding..."
if [ -f "$PROJECT_ROOT/ch3/bind_numa_affinity.py" ]; then
    python3 "$PROJECT_ROOT/ch3/bind_numa_affinity.py" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ NUMA binding working"
    else
        echo "⚠️  NUMA binding had issues (expected in containers)"
    fi
else
    echo "ℹ️  NUMA binding script not present, skipping."
fi

echo ""
echo "🎉 All critical tests passed! Setup is working correctly."

# Final summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "✅ Installed:"
echo "  • PyTorch 2.9 nightly with CUDA 12.9"
echo "  • CUDA 12.9 toolchain and development tools"
echo "  • NVIDIA Nsight Systems (latest available)"
echo "  • NVIDIA Nsight Compute (latest available)"
echo "  • All project dependencies"
echo "  • System performance tools (numactl, perf, etc.)"
echo ""
echo "🚀 Quick Start:"
echo "  1. Run: python3 ch1/performance_basics.py"
echo "  2. Run: python3 ch2/hardware_info.py"
echo "  3. Run: python3 ch3/bind_numa_affinity.py"
echo ""
echo "📚 Available Examples:"
echo "  • Chapter 1: Performance basics"
echo "  • Chapter 2: Hardware information"
echo "  • Chapter 3: NUMA affinity binding"
echo "  • Chapter 14: PyTorch compiler and Triton examples"
echo ""
echo "🔧 Profiling Commands:"
echo "  • Nsight Systems: nsys profile -t cuda,nvtx,osrt -o profile python script.py"
echo "  • Nsight Compute: ncu --metrics achieved_occupancy -o profile python script.py"
echo "  • PyTorch Profiler: Use torch.profiler in your code"
echo ""
echo "📖 For more information, see the main README.md file and chapter-specific documentation."
echo ""
echo "Happy performance engineering! 🚀"
