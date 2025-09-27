#!/usr/bin/env bash
set -euo pipefail

PROFILE_ROOT="${1:-profiles}"

if [[ ! -d "${PROFILE_ROOT}" ]]; then
  echo "No profile directory found at '${PROFILE_ROOT}'. Nothing to clean."
  exit 0
fi

read -p "This will remove '${PROFILE_ROOT}' and all generated profiler artefacts. Continue? [y/N] " reply
if [[ "${reply}" != "y" && "${reply}" != "Y" ]]; then
  echo "Aborting cleanup."
  exit 1
fi

rm -rf "${PROFILE_ROOT}"
mkdir -p "${PROFILE_ROOT}"
echo "Cleaned '${PROFILE_ROOT}'."
