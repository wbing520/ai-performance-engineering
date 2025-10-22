#!/usr/bin/env bash
# Aggregate profiler artefacts into CSV/Markdown tables for the manuscript.

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

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${CODE_ROOT}/output"

TOTAL_STEPS=4

python3 "${CODE_ROOT}/tools/extract_ncu_subset.py" "${OUTPUT_DIR}/reports/*.csv"
progress 1 "$TOTAL_STEPS" "Nsight Compute metrics extracted"

python3 "${CODE_ROOT}/tools/extract_nsys_summary.py" "${OUTPUT_DIR}/traces/*.nsys-rep"
progress 2 "$TOTAL_STEPS" "Nsight Systems summaries extracted"

python3 "${CODE_ROOT}/tools/extract_pytorch_profile.py" "${CODE_ROOT}/profiles"/*/pytorch_*/*
progress 3 "$TOTAL_STEPS" "PyTorch profiler summaries extracted"

CSV_DIR="${OUTPUT_DIR}"
MD_FILE="${OUTPUT_DIR}/manuscript_tables.md"

{
    echo "# Manuscript Metrics"
    echo
    echo "## Nsight Compute"
    echo
    if [[ -f "${CSV_DIR}/metrics_summary.csv" ]]; then
        csvtool format '|' "${CSV_DIR}/metrics_summary.csv" || cat "${CSV_DIR}/metrics_summary.csv"
    else
        echo "(Nsight Compute metrics not found)"
    fi
    echo
    echo "## Nsight Systems"
    echo
    if [[ -f "${CSV_DIR}/nsys_summary.csv" ]]; then
        csvtool format '|' "${CSV_DIR}/nsys_summary.csv" || cat "${CSV_DIR}/nsys_summary.csv"
    else
        echo "(Nsight Systems metrics not found)"
    fi
    echo
    echo "## PyTorch Profiler"
    echo
    if [[ -f "${CSV_DIR}/pytorch_profile_metadata.csv" ]]; then
        echo "### Metadata"
        csvtool format '|' "${CSV_DIR}/pytorch_profile_metadata.csv" || cat "${CSV_DIR}/pytorch_profile_metadata.csv"
    else
        echo "(PyTorch metadata not found)"
    fi
    echo
    if [[ -f "${CSV_DIR}/pytorch_profile_operators.csv" ]]; then
        echo "### Operator Summary"
        csvtool format '|' "${CSV_DIR}/pytorch_profile_operators.csv" || cat "${CSV_DIR}/pytorch_profile_operators.csv"
    else
        echo "(PyTorch operator summary not found)"
    fi
} > "${MD_FILE}"

echo "Wrote ${MD_FILE}"

progress 4 "$TOTAL_STEPS" "Extraction complete"
