#!/bin/bash
# Legacy wrapper kept for compatibility. The project now targets Blackwell (SM100) only.

set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "[info] Ignoring requested architecture '$1' – repository is Blackwell-only (sm_100)."
fi

echo "Blackwell build profile is active. No architecture switching required."
