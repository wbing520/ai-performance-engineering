#!/usr/bin/env bash
# Sanity-check profiling configuration without running heavy workloads.

set -euo pipefail

progress() {
  local current=$1
  local total=$2
  local message=$3
  local width=40
  local filled=$(( current * width / total ))
  local empty=$(( width - filled ))
  printf '\r[%s%s] %s' \
    "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
    "$(printf '%*s' "$empty" '' | tr ' ' '-')" \
    "$message"
  if (( current == total )); then
    printf '\n'
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}
TOTAL_STEPS=3

progress 1 "$TOTAL_STEPS" "Inspecting example registry"

# 1. Ensure harness modules import correctly and list examples/mappings.
${PYTHON} - <<'PY'
from scripts.example_registry import EXAMPLES
from scripts.metrics_config import resolve_overrides

print("=== Registered Examples & Chapter Tags ===")
for example in EXAMPLES:
    overrides = resolve_overrides(example)
    print(f"{example.name} :: tags={example.tags} :: profile_modes={overrides.pytorch_modes} :: ncu={len(overrides.ncu_metrics)} metrics :: nsys_trace={','.join(overrides.nsys_trace) or 'default'}")
PY

# 2. Dry-run the harness to confirm command construction without execution.
progress 2 "$TOTAL_STEPS" "Dry-running harness (max 3 examples)"
${PYTHON} "${REPO_ROOT}/scripts/profile_harness.py" --profile all --dry-run --max-examples 3 --summary

# 3. Report available extraction helpers.
progress 3 "$TOTAL_STEPS" "Reporting extraction helpers"
cat <<'INFO'

Extraction helpers:
  ./extract.sh                             # Aggregate CSV/Markdown tables
  python tools/extract_ncu_subset.py ...   # Nsight Compute metrics
  python tools/extract_nsys_summary.py ... # Nsight Systems summaries
  python tools/extract_pytorch_profile.py  # Torch profiler summaries

INFO
