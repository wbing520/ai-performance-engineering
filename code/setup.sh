#!/usr/bin/env bash

# AI Performance Engineering CUDA 13 setup helper
# Installs CUDA 13 toolchain and PyTorch 2.9 (cu130) nightly with Triton 3.5
# Requires Ubuntu 20.04/22.04+ and root privileges

set -euo pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -x
fi

log_step() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ensure we are root because apt and system installs require it
if [[ "${EUID}" -ne 0 ]]; then
    echo "❌ This script must be run as root (try sudo)."
    exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export PIP_BREAK_SYSTEM_PACKAGES=1

log_step "Updating apt package index"
apt-get update

log_step "Installing core system dependencies"
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    lsb-release \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    software-properties-common \
    wget

UBUNTU_RELEASE="$(lsb_release -rs)"
UBUNTU_ID="${UBUNTU_RELEASE//./}"        # e.g. 22.04 -> 2204
CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_ID}/x86_64/"
CUDA_KEYRING="/usr/share/keyrings/cuda-archive-keyring.gpg"
CUDA_APT_SOURCE="/etc/apt/sources.list.d/cuda-ubuntu${UBUNTU_ID}-x86_64.list"
CUDA_APT_SOURCE_LEGACY="/etc/apt/sources.list.d/cuda-ubuntu${UBUNTU_ID}.list"

log_step "Configuring NVIDIA CUDA 13 apt repository for ubuntu${UBUNTU_ID}"
tmp_keyring="$(mktemp)"
if ! curl -fsSL "${CUDA_REPO_URL}cuda-archive-keyring.gpg" -o "${tmp_keyring}"; then
    echo "❌ Failed to download CUDA repository key from ${CUDA_REPO_URL}"
    echo "    Ensure NVIDIA has published CUDA 13 for your Ubuntu release or adjust CUDA version."
    rm -f "${tmp_keyring}"
    exit 1
fi
install -Dm644 "${tmp_keyring}" "${CUDA_KEYRING}"
rm -f "${tmp_keyring}"

if [[ -f "${CUDA_APT_SOURCE_LEGACY}" && ! -L "${CUDA_APT_SOURCE_LEGACY}" ]]; then
    log_step "Removing legacy CUDA apt source ${CUDA_APT_SOURCE_LEGACY}"
    rm -f "${CUDA_APT_SOURCE_LEGACY}"
fi

if [[ ! -f "${CUDA_APT_SOURCE}" ]] || ! grep -q "${CUDA_REPO_URL}" "${CUDA_APT_SOURCE}"; then
    cat > "${CUDA_APT_SOURCE}" <<EOF
deb [signed-by=${CUDA_KEYRING}] ${CUDA_REPO_URL} /
EOF
else
    log_step "CUDA apt source already points to ${CUDA_REPO_URL}"
fi

log_step "Refreshing apt indices with CUDA 13 packages"
apt-get update

CUDA_PACKAGES=(
    cuda-toolkit-13-0
    cuda-command-line-tools-13-0
    cuda-sanitizer-13-0
    cuda-nsight-compute-13-0
    cuda-nsight-systems-13-0
)

log_step "Installing CUDA 13 toolchain and tooling"
for pkg in "${CUDA_PACKAGES[@]}"; do
    if apt-cache show "${pkg}" >/dev/null 2>&1; then
        apt-get install -y "${pkg}"
    else
        log_step "Package ${pkg} not available in apt repo (skipping)"
    fi
done

if ! command -v nvcc >/dev/null 2>&1; then
    echo "❌ nvcc not found after installation. Check CUDA repo availability and rerun."
    exit 1
fi

log_step "Upgrading pip tooling"
python3 -m pip install --upgrade pip setuptools wheel

log_step "Ensuring blinker can be upgraded cleanly"
python3 -m pip install --ignore-installed blinker

log_step "Installing PyTorch 2.9 (cu130) nightly, Triton 3.5, and project dependencies"
python3 -m pip install --upgrade --pre -r "${SCRIPT_DIR}/requirements_latest.txt"

log_step "Verifying CUDA, PyTorch, and Triton versions"
python3 - <<'PYTHON'
import shutil
import torch

try:
    import triton
except ImportError:
    triton = None

print(f"nvcc path: {shutil.which('nvcc')}")
print(f"torch version: {torch.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch cuda device: {torch.cuda.get_device_name(0)}")
print(f"triton version: {getattr(triton, '__version__', 'not installed')}")
PYTHON

log_step "CUDA 13 & PyTorch 2.9 environment ready. You can now run examples under ${SCRIPT_DIR}"
