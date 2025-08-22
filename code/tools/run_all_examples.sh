#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="$ROOT_DIR/run_report_${TS}.md"

echo "# Run Report ($TS)" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

echo "## System" >> "$REPORT_FILE"
echo "- Host: $(hostname)" >> "$REPORT_FILE"
echo "- GPUs:" >> "$REPORT_FILE"
nvidia-smi -L | sed 's/^/- /' >> "$REPORT_FILE" || true
echo "" >> "$REPORT_FILE"

# Detect Blackwell (sm_100) capability using nvidia-smi name
IS_BLACKWELL=0
if nvidia-smi -L 2>/dev/null | grep -q "B200\|B300"; then
  IS_BLACKWELL=1
fi

echo "## Build" >> "$REPORT_FILE"

build_dirs=(
  "$ROOT_DIR/ch6"
  "$ROOT_DIR/ch7"
  "$ROOT_DIR/ch8"
  "$ROOT_DIR/ch9"
  "$ROOT_DIR/ch10"
  "$ROOT_DIR/ch11"
  "$ROOT_DIR/ch12"
  "$ROOT_DIR/ch2"
)

for d in "${build_dirs[@]}"; do
  if [ -f "$d/Makefile" ]; then
    if [ "$IS_BLACKWELL" = "1" ]; then
      echo "- Building $d (ARCH=sm_100)" | tee -a "$REPORT_FILE"
      (cd "$d" && make clean >/dev/null 2>&1 || true)
      (cd "$d" && make ARCH=sm_100 | sed 's/^/  /') >> "$REPORT_FILE" 2>&1 || true
    else
      echo "- Building $d (ARCH=sm_90)" | tee -a "$REPORT_FILE"
      (cd "$d" && make clean >/dev/null 2>&1 || true)
      (cd "$d" && make ARCH=sm_90 | sed 's/^/  /') >> "$REPORT_FILE" 2>&1 || true
    fi
  fi
done

echo "" >> "$REPORT_FILE"
echo "## Run CUDA Binaries" >> "$REPORT_FILE"

run_bin_dir() {
  local dir="$1"
  if [ ! -d "$dir" ]; then return; fi
  while IFS= read -r -d '' exe; do
    base="$(basename "$exe")"
    # Skip Blackwell-only binaries on non-Blackwell
    if [ "$IS_BLACKWELL" = "0" ] && echo "$base" | grep -q "_blackwell"; then
      continue
    fi
    echo "- Running $exe" | tee -a "$REPORT_FILE"
    if timeout 60s "$exe" >"$exe.out" 2>"$exe.err"; then
      echo "  - status: OK" >> "$REPORT_FILE"
      echo "  - output (tail):" >> "$REPORT_FILE"
      tail -n 10 "$exe.out" | sed 's/^/    /' >> "$REPORT_FILE"
    else
      echo "  - status: FAIL" >> "$REPORT_FILE"
      echo "  - stderr (tail):" >> "$REPORT_FILE"
      tail -n 30 "$exe.err" | sed 's/^/    /' >> "$REPORT_FILE"
    fi
  done < <(find "$dir" -maxdepth 1 -type f -perm -111 -print0)
}

for d in "${build_dirs[@]}"; do
  run_bin_dir "$d"
done

echo "" >> "$REPORT_FILE"
echo "## Run Python examples (selected)" >> "$REPORT_FILE"

run_py() {
  local script="$1"
  if [ "$IS_BLACKWELL" = "0" ] && echo "$script" | grep -q "blackwell"; then
    return
  fi
  echo "- $script" >> "$REPORT_FILE"
  if timeout 600s python "$script" >"$script.out" 2>"$script.err"; then
    echo "  - status: OK" >> "$REPORT_FILE"
    tail -n 30 "$script.out" | sed 's/^/    /' >> "$REPORT_FILE"
  else
    echo "  - status: FAIL/WARN" >> "$REPORT_FILE"
    tail -n 50 "$script.err" | sed 's/^/    /' >> "$REPORT_FILE"
  fi
}

run_py "$ROOT_DIR/ch14/torch_compiler_examples.py"
run_py "$ROOT_DIR/ch2/hardware_info.py"

echo "" >> "$REPORT_FILE"
echo "Report written to: $REPORT_FILE"
echo "$REPORT_FILE"


