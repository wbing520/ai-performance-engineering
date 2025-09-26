#!/bin/bash

# Blackwell-only build workflow for AI Performance Engineering
# Targets: NVIDIA Blackwell B200/B300 (SM100) with CUDA 12.8, PyTorch 2.8, Triton 3.3

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}
ARCH="sm_100"
REQ_FILE="$SCRIPT_DIR/requirements_latest.txt"

section() {
  echo
  echo "=== $1 ==="
}

section "Blackwell Build Pipeline"
echo "Repository  : $REPO_ROOT"
echo "Python      : $(command -v "$PYTHON" || echo 'not found')"
echo "CUDA target : $ARCH (Blackwell B200/B300)"

section "Environment Versions"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | head -n 4
else
  echo "⚠ nvcc not found (CUDA compilation will fail)"
fi

if "$PYTHON" - <<'PY' 2>/dev/null; then
import torch, triton
print(f"PyTorch    : {torch.__version__}")
print(f"Triton     : {triton.__version__}")
PY
  :
else
  echo "⚠ Unable to import torch/triton"
fi

section "Installing Python dependencies"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "❌ requirements file not found: $REQ_FILE" >&2
  exit 1
fi

TMP_REQ=$(mktemp)

cp "$REQ_FILE" "$TMP_REQ"

pip install --upgrade --no-cache-dir -r "$TMP_REQ"
rm -f "$TMP_REQ"

section "Verifying architecture helpers"
"$PYTHON" - <<'PY'
import sys
sys.path.append('code')
from arch_config import arch_config, configure_optimizations
arch_config.print_info()
configure_optimizations()
print("✓ Architecture helpers configured for Blackwell")
PY

section "Building CUDA examples"
pushd "$SCRIPT_DIR" >/dev/null
find "$SCRIPT_DIR" -name Makefile -print | while read -r makefile; do
  make_dir="$(dirname "$makefile")"
  target_name="$(basename "$make_dir")"
  echo "-- Building $make_dir"
  pushd "$make_dir" >/dev/null
  make clean >/dev/null 2>&1 || true
  if make ARCH=$ARCH; then
    if [[ -f "$target_name" ]]; then
      echo "   ✓ $target_name built"
    fi
  else
    echo "   ⚠ Build failed in $make_dir" >&2
  fi
  popd >/dev/null
  echo
done
popd >/dev/null

section "Python syntax checks"
find "$SCRIPT_DIR" -path "*/ch*/*" -name "*.py" -print | while read -r pyfile; do
  case "$(basename "$pyfile")" in
    test_*|*_test.py) continue ;;
    arch_config.py) continue ;;
  esac
  if "$PYTHON" -m py_compile "$pyfile" 2>/dev/null; then
    echo "✓ $(realpath --relative-to="$SCRIPT_DIR" "$pyfile")"
  else
    echo "⚠ Syntax error: $pyfile" >&2
  fi
done

section "Done"
echo "Blackwell build pipeline completed."
