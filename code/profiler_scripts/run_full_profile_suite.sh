#!/bin/bash
# Run Nsight Systems/Compute + auxiliary profilers across key chapter workloads.
# Generates per-run CSV exports for book/table comparison and reports success/fail status.

set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# List of profiling tasks: format "label|script_path|tools"
# tools is the comma-separated list understood by master_profile.sh (e.g., "nsys,ncu" or "all").
declare -A TOOL_OVERRIDES
# Example overrides (add as needed):
# TOOL_OVERRIDES["code/ch4/after_reinit_comm.py"]="nsys"
# TOOL_OVERRIDES["code/ch4/dist_allreduce.py"]="nsys,ncu,hta"
# TOOL_OVERRIDES["code/ch7/vectorized_pytorch.py"]="nsys,pytorch"

declare -A SKIP_SCRIPTS
# Example skip:
# SKIP_SCRIPTS["code/ch13/train_deepseek_v3.py"]=1

TASKS=()
while IFS= read -r script; do
  rel_path=${script#$REPO_ROOT/}
  chapter_dir=$(basename "$(dirname "$rel_path")")
  [[ $chapter_dir =~ ^ch[0-9]+$ ]] || continue
  [[ -n ${SKIP_SCRIPTS[$rel_path]:-} ]] && continue
  base=$(basename "$rel_path")
  name_no_ext=${base%.*}
  chapter_num=${chapter_dir#ch}
  printf -v label "Ch%02d-%s" "$chapter_num" "$name_no_ext"

  if [[ -n ${TOOL_OVERRIDES[$rel_path]:-} ]]; then
    tools=${TOOL_OVERRIDES[$rel_path]}
  else
    # Default tool selection based on suffix, with chapter-specific tweaks
    case "$rel_path" in
      code/ch4/*)
        tools="nsys,ncu"  # focus on overlap/SM counters
        ;;
      code/ch7/*)
        case "${rel_path##*.}" in
          py) tools="nsys,ncu,pytorch" ;;
          cu) tools="nsys,ncu" ;;
          sh) tools="nsys" ;;
          *) tools="nsys" ;;
        esac
        ;;
      code/ch8/*)
        case "${rel_path##*.}" in
          py) tools="nsys,ncu,pytorch" ;;
          cu) tools="nsys,ncu" ;;
          sh) tools="nsys" ;;
          *) tools="nsys" ;;
        esac
        ;;
      code/ch9/*)
        case "${rel_path##*.}" in
          py) tools="nsys,ncu,pytorch" ;;
          cu) tools="nsys,ncu" ;;
          sh) tools="nsys" ;;
          *) tools="nsys" ;;
        esac
        ;;
      code/ch13/*|code/ch14/*|code/ch16/*)
        case "${rel_path##*.}" in
          py) tools="nsys,ncu,pytorch" ;;
          cu) tools="nsys,ncu" ;;
          sh) tools="nsys" ;;
          *) tools="nsys" ;;
        esac
        ;;
      code/ch18/*|code/ch19/*)
        case "${rel_path##*.}" in
          py) tools="nsys,ncu" ;;
          cu) tools="nsys,ncu" ;;
          sh) tools="nsys" ;;
          *) tools="nsys" ;;
        esac
        ;;
      *)
        case "${rel_path##*.}" in
          py) tools="nsys,ncu,pytorch" ;;
          cu) tools="nsys,ncu" ;;
          sh) tools="nsys" ;;
          *) tools="nsys" ;;
        esac
        ;;
    esac
  fi

  TASKS+=("$label|$rel_path|$tools")
done < <(find "$REPO_ROOT/code" -maxdepth 2 -type f \( -name '*.py' -o -name '*.cu' -o -name '*.sh' \) ! -name '*.pyc' | sort)

LOG_DIR="$REPO_ROOT/profile_runs"
mkdir -p "$LOG_DIR"

if (( ${#TASKS[@]} == 0 )); then
  echo "No chapter scripts discovered under code/ch*/"
  exit 1
fi

echo "Discovered ${#TASKS[@]} chapter scripts to profile."

ERRORS=()

run_and_export() {
  local label="$1"
  local script_path="$2"
  local tools="$3"

  local abs_script="$REPO_ROOT/$script_path"
  local log_file="$LOG_DIR/${label}_$(date +%Y%m%d_%H%M%S).log"

  echo "=== [$label] Profiling $script_path ($tools) ===" | tee "$log_file"

  if [[ ! -f "$abs_script" ]]; then
    echo "ERROR: script not found: $abs_script" | tee -a "$log_file"
    ERRORS+=("$label: script not found")
    return
  fi

  local before_dirs
  before_dirs=$(ls -1d master_profile_* 2>/dev/null || true)

  (cd "$REPO_ROOT" && bash "$SCRIPT_DIR/master_profile.sh" "$script_path" auto "$tools") >>"$log_file" 2>&1
  local exit_code=$?

  if [[ $exit_code -ne 0 ]]; then
    echo "ERROR: profiling failed for $label (exit $exit_code). See $log_file" | tee -a "$log_file"
    ERRORS+=("$label: master_profile exit $exit_code")
    return
  fi

  local latest_dir
  latest_dir=$(cd "$REPO_ROOT" && ls -1dt master_profile_* 2>/dev/null | head -n 1)

  if [[ -z "$latest_dir" ]]; then
    echo "WARNING: No master_profile output detected for $label" | tee -a "$log_file"
    ERRORS+=("$label: no output dir")
    return
  fi

  local abs_latest="$REPO_ROOT/$latest_dir"
  echo "Output directory: $latest_dir" | tee -a "$log_file"

  # Export Nsight Systems CSVs
  for rep in "$abs_latest"/profile_nsys_*/*.nsys-rep; do
    [[ -f "$rep" ]] || continue
    local csv_base="${rep%.nsys-rep}"
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$rep" > "${csv_base}_cuda_gpu_kern_sum.csv" 2>>"$log_file" || \ 
      echo "WARNING: Failed to export Nsight Systems report for $rep" | tee -a "$log_file"
  done

  # Export Nsight Compute CSVs
  for rep in "$abs_latest"/profile_ncu_*/*.ncu-rep; do
    [[ -f "$rep" ]] || continue
    local csv_base="${rep%.ncu-rep}"
    ncu --import "$rep" --csv > "${csv_base}.csv" 2>>"$log_file" || \ 
      echo "WARNING: Failed to export Nsight Compute report for $rep" | tee -a "$log_file"
  done

  echo "Completed $label" | tee -a "$log_file"
}

for entry in "${TASKS[@]}"; do
  IFS='|' read -r label script tools <<<"$entry"
  run_and_export "$label" "$script" "$tools"
  echo
  sleep 2
done

if (( ${#ERRORS[@]} > 0 )); then
  echo "=== Completed with warnings/errors ==="
  printf ' - %s
' "${ERRORS[@]}"
  exit 1
fi

echo "All profiling tasks completed successfully. CSV exports are in the generated master_profile_* directories."
