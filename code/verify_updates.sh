#!/bin/bash
# Blackwell-only consistency checks for AI Performance Engineering

set -euo pipefail

SECTION_COUNTER=0
section() {
  SECTION_COUNTER=$((SECTION_COUNTER + 1))
  echo
  echo "[$SECTION_COUNTER] $1"
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

section "Environment"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
else
  echo "⚠ nvidia-smi not available"
fi

section "Makefile sanity"
ISSUES=0
while IFS= read -r makefile; do
  rel_path=${makefile#"$ROOT_DIR/"}
  if ! grep -q "CUDA_VERSION = 12.8" "$makefile"; then
    echo "❌ $rel_path missing 'CUDA_VERSION = 12.8'"
    ISSUES=$((ISSUES + 1))
  fi
  if grep -q "sm_90" "$makefile"; then
    echo "❌ $rel_path still references sm_90"
    ISSUES=$((ISSUES + 1))
  fi
  if ! grep -q '\-arch=\$(ARCH)' "$makefile"; then
    echo "❌ $rel_path missing parametrised -arch=\$(ARCH) flag"
    ISSUES=$((ISSUES + 1))
  fi
  if ! grep -q "Blackwell" "$makefile"; then
    echo "⚠ $rel_path has no Blackwell status message"
  fi
  if grep -q "profile-hta" "$makefile"; then
    :
  else
    echo "⚠ $rel_path lacks profile-hta target"
  fi
  if grep -q "profile-perf" "$makefile"; then
    :
  else
    echo "⚠ $rel_path lacks profile-perf target"
  fi
  if grep -q "profile-all" "$makefile"; then
    :
  else
    echo "⚠ $rel_path lacks profile-all target"
  fi
  if grep -q "Hopper" "$makefile"; then
    echo "❌ $rel_path still references Hopper"
    ISSUES=$((ISSUES + 1))
  fi
done < <(find "$ROOT_DIR" -name Makefile -type f ! -path "*/archive/*")

section "Requirements"
REQ_MD5=$(md5sum "$ROOT_DIR/requirements_latest.txt" | cut -d' ' -f1)
echo "requirements_latest.txt md5: $REQ_MD5"
if grep -q "sm_90" "$ROOT_DIR/requirements_latest.txt"; then
  echo "❌ requirements file references legacy sm_90 entries"
  ISSUES=$((ISSUES + 1))
fi
if ! grep -q "tokenizers==0.19.1" "$ROOT_DIR/requirements_latest.txt"; then
  echo "❌ tokenizers pin incorrect"
  ISSUES=$((ISSUES + 1))
fi
if ! grep -q "scikit-learn==1.4.2" "$ROOT_DIR/requirements_latest.txt"; then
  echo "❌ scikit-learn pin incorrect"
  ISSUES=$((ISSUES + 1))
fi

section "Code references"
if rg --glob "!**/archive/**" --glob "!**/verify_updates.sh" --glob "!**/update_cuda_versions.sh" --glob "!**/update_architecture_switching.sh" -q "Hopper" "$ROOT_DIR"; then
  echo "❌ Legacy architecture references remain in code base"
  ISSUES=$((ISSUES + 1))
else
  echo "✓ No legacy architecture references detected"
fi
if rg --glob "!**/archive/**" --glob "!**/verify_updates.sh" --glob "!**/update_cuda_versions.sh" --glob "!**/update_architecture_switching.sh" -q "sm_90" "$ROOT_DIR"; then
  echo "❌ sm_90 references remain"
  ISSUES=$((ISSUES + 1))
else
  echo "✓ No sm_90 references detected"
fi

section "Summary"
if [[ $ISSUES -eq 0 ]]; then
  echo "✓ Blackwell-only checks passed"
else
  echo "⚠ Found $ISSUES blocking issues"
  exit 1
fi
