#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS="${SCRIPT_DIR}/profile_harness.py"
PYTORCH_RUNNER="${SCRIPT_DIR}/pytorch_profiler_runner.py"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT_ROOT="${REPO_ROOT}/profiles/manual"
PYTHON_DEFAULT="${PYTHON:-python}"

print_usage() {
    cat <<'USAGE'
Usage:
  enhanced_profiling.sh --list
  enhanced_profiling.sh [HARNESS_FLAGS...]
  enhanced_profiling.sh <script.py> [--arch sm_100] [--tool nsys|ncu|pytorch|hta|perf|all]
                         [--pytorch-mode full] [--output-root DIR] [--python PYTHON]
                         [-- script-args ...]

Harness mode (preferred):
  For registered examples, forward directly to profile_harness.py. Any
  of the harness arguments (--examples, --tags, --profile, --profile-mode,
  --dry-run, --skip-existing, etc.) can be passed through.

Direct mode:
  Executes a specific script path with Nsight Systems, Nsight Compute,
  PyTorch profiler, HTA, and/or perf. Optional script arguments can be
  provided after a literal "--".

Examples:
  enhanced_profiling.sh --profile nsys --examples ch14_triton_examples
  enhanced_profiling.sh code/ch7/memory_access_pytorch.py --tool ncu
  enhanced_profiling.sh code/ch9/fusion_pytorch.py --tool pytorch --pytorch-mode memory -- --batch-size 4
USAGE
}

require_command() {
    local cmd="$1"
    local label="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "⚠ Skipping ${label}; '${cmd}' not found in PATH" >&2
        return 1
    fi
    return 0
}

print_command() {
    local parts=("$@")
    printf '→ %s\n' "$(printf '%q ' "${parts[@]}")"
}

timestamp() {
    date +%Y%m%d_%H%M%S
}

resolve_path() {
    local target="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath "$target"
    else
        python - "${target}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
    fi
}

resolve_architecture() {
    local requested="$1"
    if [[ "$requested" != "auto" ]]; then
        echo "$requested"
        return
    fi

    local detected="sm_100"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 || true)
        if [[ -n "$gpu_name" && ! "$gpu_name" =~ (B200|B300) ]]; then
            echo "sm_100"
            echo "⚠ Non-Blackwell GPU detected (${gpu_name}); defaulting to sm_100" >&2
            return
        fi
    else
        echo "⚠ Unable to query GPU via nvidia-smi; assuming sm_100" >&2
    fi
    echo "$detected"
}

run_nsys() {
    require_command nsys "Nsight Systems" || return
    local base="${SESSION_DIR}/nsys_${ARCH_VALUE}_$(timestamp)"
    local cmd=(nsys profile --force-overwrite=true -o "$base" -t cuda,nvtx,osrt,cudnn,cublas \
        -s cpu --python-sampling=true --python-sampling-frequency=1000 \
        --cudabacktrace=true --stats=true \
        "$PYTHON_BIN" "$SCRIPT_PATH")
    if ((${#SCRIPT_ARGS[@]})); then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Nsight Systems report: ${base}.nsys-rep"
}

run_ncu() {
    require_command ncu "Nsight Compute" || return
    pkill -f nsys >/dev/null 2>&1 || true
    local base="${SESSION_DIR}/ncu_${ARCH_VALUE}_$(timestamp)"
    local cmd=(ncu --set full -o "$base" "$PYTHON_BIN" "$SCRIPT_PATH")
    if ((${#SCRIPT_ARGS[@]})); then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Nsight Compute report: ${base}.ncu-rep"
}

run_hta() {
    require_command nsys "Nsight Systems" || return
    local base="${SESSION_DIR}/hta_${ARCH_VALUE}_$(timestamp)"
    local cmd=(nsys profile --force-overwrite=true -o "$base" -t cuda,nvtx,osrt,cudnn,cublas,nccl \
        -s cpu --python-sampling=true --python-sampling-frequency=1000 --cudabacktrace=true \
        --stats=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop --capture-range-op=both \
        --multi-gpu=all "$PYTHON_BIN" "$SCRIPT_PATH")
    if ((${#SCRIPT_ARGS[@]})); then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ HTA report: ${base}.nsys-rep"
}

run_perf() {
    require_command perf "perf" || return
    local data_file="${SESSION_DIR}/perf_${ARCH_VALUE}_$(timestamp).data"
    local cmd=(perf record --call-graph dwarf -o "$data_file" "$PYTHON_BIN" "$SCRIPT_PATH")
    if ((${#SCRIPT_ARGS[@]})); then
        cmd+=("${SCRIPT_ARGS[@]}")
    fi
    print_command "${cmd[@]}"
    "${cmd[@]}"
    echo "  ↳ Perf data captured: ${data_file}"
    echo "     View with: perf report -i ${data_file}"
}

run_pytorch() {
    local modes=("${PYTORCH_MODES[@]}")
    if ((${#modes[@]} == 0)); then
        modes=(full)
    fi
    for mode in "${modes[@]}"; do
        local out_dir="${SESSION_DIR}/pytorch_${mode}"
        local cmd=("$PYTHON_BIN" "$PYTORCH_RUNNER" "$SCRIPT_PATH" --output-dir "$out_dir" --profile-mode "$mode")
        if ((${#SCRIPT_ARGS[@]})); then
            cmd+=(--script-args "${SCRIPT_ARGS[@]}")
        fi
        print_command "${cmd[@]}"
        "${cmd[@]}"
        echo "  ↳ PyTorch profiler output: ${out_dir}"
    done
}

run_comprehensive() {
    run_nsys
    run_ncu
    run_pytorch
    run_hta
    run_perf
}

if (( $# == 0 )); then
    print_usage
    exit 1
fi

case "$1" in
    --help|-h)
        print_usage
        exit 0
        ;;
    --list)
        "$PYTHON_DEFAULT" "$HARNESS" --list
        exit $?
        ;;
    --examples|--example|--tags|--tag|--profile|--profile-mode|--output-root|--dry-run|--skip-existing|--max-examples)
        "$PYTHON_DEFAULT" "$HARNESS" "$@"
        exit $?
        ;;
esac

SCRIPT_PATH="$1"
shift

if [[ ! -f "$SCRIPT_PATH" ]]; then
    if [[ -f "${REPO_ROOT}/${SCRIPT_PATH}" ]]; then
        SCRIPT_PATH="${REPO_ROOT}/${SCRIPT_PATH}"
    else
        echo "✗ Unable to locate script: ${SCRIPT_PATH}" >&2
        exit 1
    fi
fi

SCRIPT_PATH="$(resolve_path "$SCRIPT_PATH")"
SCRIPT_BASENAME="$(basename "$SCRIPT_PATH")"
SCRIPT_ARGS=()
ARCH="auto"
PROFILE_SPEC="all"
PYTORCH_MODES=()
PYTHON_BIN="${PYTHON:-python}"
# Allow overriding via environment but default to earlier value
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
ARCH_SET=false
PROFILE_SET=false

while (( $# > 0 )); do
    case "$1" in
        --arch)
            ARCH="$2"
            ARCH_SET=true
            shift 2
            ;;
        --tool|--profile-type|--profile)
            if $PROFILE_SET; then
                PROFILE_SPEC+="${PROFILE_SPEC:+,}$2"
            else
                PROFILE_SPEC="$2"
                PROFILE_SET=true
            fi
            shift 2
            ;;
        --pytorch-mode)
            PYTORCH_MODES+=("$2")
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --)
            shift
            SCRIPT_ARGS=("$@")
            break
            ;;
        *)
            if ! $ARCH_SET; then
                ARCH="$1"
                ARCH_SET=true
            elif ! $PROFILE_SET; then
                PROFILE_SPEC="$1"
                PROFILE_SET=true
            else
                echo "✗ Unknown argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
    [[ $# -eq 0 ]] && break
done

ARCH_VALUE="$(resolve_architecture "$ARCH")"
mkdir -p "$OUTPUT_ROOT"
SESSION_DIR="${OUTPUT_ROOT}/$(timestamp)_${SCRIPT_BASENAME%.*}"
mkdir -p "$SESSION_DIR"

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export CUDA_CACHE_DISABLE=${CUDA_CACHE_DISABLE:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

if [[ "$SCRIPT_PATH" =~ /code/ch1/ || "$SCRIPT_PATH" =~ /code/ch2/ || "$SCRIPT_PATH" =~ /code/ch1[3-9]/ || "$SCRIPT_PATH" =~ /code/ch20/ ]]; then
    export TORCHINDUCTOR_AUTOTUNE=${TORCHINDUCTOR_AUTOTUNE:-0}
    export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
fi


PROFILE_SPEC="${PROFILE_SPEC,,}"
IFS="," read -r -a RAW_TOOLS <<< "$PROFILE_SPEC"
if ((${#RAW_TOOLS[@]} == 0)); then
    RAW_TOOLS=("all")
fi
TOOLS=()
for tool in "${RAW_TOOLS[@]}"; do
    case "$tool" in
        all)
            TOOLS=(nsys ncu pytorch hta perf)
            break
            ;;
        nsys|ncu|hta|perf|pytorch|torch)
            norm="$tool"
            [[ "$norm" == "torch" ]] && norm="pytorch"
            TOOLS+=("${norm}")
            ;;
        *)
            echo "✗ Unknown profile tool: ${tool}" >&2
            exit 1
            ;;
    esac
done

# Deduplicate while preserving order
DEDUP_TOOLS=()
for tool in "${TOOLS[@]}"; do
    seen=false
    for existing in "${DEDUP_TOOLS[@]}"; do
        if [[ "$existing" == "$tool" ]]; then
            seen=true
            break
        fi
    done
    if [[ "$seen" == false ]]; then
        DEDUP_TOOLS+=("$tool")
    fi
done
TOOLS=("${DEDUP_TOOLS[@]}")

cat <<SUMMARY
=== Enhanced Profiling ===
Script       : ${SCRIPT_PATH}
Python       : ${PYTHON_BIN}
Architecture : ${ARCH_VALUE}
Tools        : ${TOOLS[*]}
Output Dir   : ${SESSION_DIR}
SUMMARY
if ((${#SCRIPT_ARGS[@]})); then
    printf 'Arguments    : %s\n' "$(printf '%q ' "${SCRIPT_ARGS[@]}")"
fi

declare -A TOOL_RUNNERS=(
    [nsys]=run_nsys
    [ncu]=run_ncu
    [hta]=run_hta
    [perf]=run_perf
    [pytorch]=run_pytorch
)

for tool in "${TOOLS[@]}"; do
    "${TOOL_RUNNERS[$tool]}"
done

echo
echo "Profiling complete. Artifacts available under: ${SESSION_DIR}"
