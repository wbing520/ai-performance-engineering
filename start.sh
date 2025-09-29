#!/usr/bin/env bash
# Launch the profiling harness across all registered chapters using the
# chapter-specific metric overrides defined in metrics_config.py.
#
# The harness runs in the background under nohup so long-running Nsight
# sessions survive the terminal. Logs and PID files live under
# ./profile_runs/harness. Pass additional harness arguments to tailor a run.

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
HARNESS="$REPO_ROOT/scripts/profile_harness.py"
SESSION_ROOT="$REPO_ROOT/profile_runs/harness"
mkdir -p "$SESSION_ROOT"

TOTAL_STEPS=3

PYTHON_BIN=${PYTHON:-}
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "ERROR: No Python interpreter found. Set \$PYTHON or install python3." >&2
    exit 1
  fi
fi

if [[ ! -f "$HARNESS" ]]; then
  echo "ERROR: Unable to locate profiling harness at $HARNESS" >&2
  exit 1
fi

progress 1 "$TOTAL_STEPS" "Using interpreter: $PYTHON_BIN"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$SESSION_ROOT/harness_${TIMESTAMP}.log"
PID_FILE="$SESSION_ROOT/harness_${TIMESTAMP}.pid"
LATEST_LOG="$SESSION_ROOT/latest.log"
LATEST_PID="$SESSION_ROOT/latest.pid"

CMD=(env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$HARNESS")
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
else
  CMD+=("--profile" "all")
fi

{
  echo "=== Profiling harness launched at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "Repository : $REPO_ROOT"
  echo "Interpreter: $PYTHON_BIN"
  echo "Command    : ${CMD[*]}"
  echo "Log file   : $LOG_FILE"
  echo "PID file   : $PID_FILE"
} > "$LOG_FILE"

cd "$REPO_ROOT"
"${CMD[@]}"
