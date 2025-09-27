#!/bin/bash
# Terminate active profiling jobs launched via profile_harness/master_profile and
# any lingering Nsight or PyTorch Inductor workers.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
LOG_ROOT="$REPO_ROOT/profile_runs/nohup"

found=0
if [[ -d "$LOG_ROOT" ]]; then
  shopt -s nullglob
  for pid_file in "$LOG_ROOT"/*.pid; do
    [[ -f "$pid_file" ]] || continue
    pid=$(<"$pid_file")
    if [[ -z "$pid" ]]; then
      echo "Skipping empty PID file: $pid_file"
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping profiling process PID $pid (from $pid_file)"
      kill "$pid" || true
      found=1
    else
      echo "Process $pid (from $pid_file) is not running"
    fi
  done
  shopt -u nullglob
fi

if command -v pkill >/dev/null 2>&1; then
  if pkill -f profile_harness.py >/dev/null 2>&1; then
    echo "Issued pkill for profile_harness.py"
    found=1
  fi
  if pkill -f master_profile.sh >/dev/null 2>&1; then
    echo "Issued pkill for master_profile.sh"
    found=1
  fi
  if pkill -f 'torch/_inductor/compile_worker' >/dev/null 2>&1; then
    echo "Issued pkill for torch/_inductor/compile_worker"
    found=1
  fi
  if pkill -f '/opt/nvidia/nsight-compute' >/dev/null 2>&1; then
    echo "Issued pkill for Nsight Compute"
    found=1
  fi
  if pkill -f '/opt/nvidia/nsight-systems' >/dev/null 2>&1 || pkill -x nsys >/dev/null 2>&1; then
    echo "Issued pkill for Nsight Systems"
    found=1
  fi
  if pkill -x perf >/dev/null 2>&1; then
    echo "Issued pkill for perf"
    found=1
  fi
fi

if [[ $found -eq 0 ]]; then
  echo "No active profiling processes found."
fi
