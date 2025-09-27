#!/bin/bash
# Launch the profiling harness for all registered examples in the background
# using nohup, record the PID, and tail the log immediately.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON:-$(command -v python3 || command -v python)}

if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: Unable to find a Python interpreter. Set \$PYTHON or install python3." >&2
  exit 1
fi

mkdir -p "$REPO_ROOT/profile_runs/nohup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$REPO_ROOT/profile_runs/nohup/profile_harness_${TIMESTAMP}.log"
PID_FILE="$REPO_ROOT/profile_runs/nohup/profile_harness_${TIMESTAMP}.pid"

CMD=(env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$SCRIPT_DIR/profile_harness.py")
if [[ $# -gt 0 ]]; then
  CMD+=("$@")
else
  CMD+=("--profile" "all")
fi

{
  echo "=== Profiling harness launched at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "Repository: $REPO_ROOT"
  echo "Command: ${CMD[*]}"
  echo "Log file: $LOG_FILE"
  echo "PID file: $PID_FILE"
} > "$LOG_FILE"

cd "$REPO_ROOT"
nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Started profiling harness in background (PID $PID)."
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Tailing log (Ctrl+C to stop tail; process continues running)..."
tail -n +1 -f "$LOG_FILE"
