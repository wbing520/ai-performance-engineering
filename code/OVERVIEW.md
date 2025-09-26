# 🚀 Overview – Blackwell-Only Stack

This repository now targets a single architecture: **NVIDIA Blackwell B200/B300 (SM100)**. Every script, kernel, and document assumes CUDA 12.8, PyTorch 2.8, and Triton 3.3. Legacy compute capability 9.x support has been removed to keep the code base lean and focused.

## 🧱 Core Components
- **Architecture configuration** lives in `arch_config.py` and always resolves to Blackwell.
- **Build orchestration** is handled by `build_all.sh`, which compiles kernels with `sm_100` and validates Python syntax.
- **Requirements** across chapters are normalised to `requirements_latest.txt` (mirrored into the per-chapter files).

## 🛠️ Toolchain Expectations
| Component | Version / Channel | Notes |
|-----------|------------------|-------|
| CUDA Toolkit | 12.8 (nvcc 12.8.93) | `nvcc -arch=sm_100` is the default everywhere. |
| PyTorch | 2.8.0 (cu128 nightly) | Install from `https://download.pytorch.org/whl/nightly/cu128`. |
| Triton | 3.3.1 | Required for Triton kernels in Chapters 14 & 16. |
| Nsight Systems | 2024.6+ | Used in profiling scripts. |
| Nsight Compute | 2024.3+ | Kernel-level analysis. |

## 📦 Installing Python Dependencies
```bash
pip install -r requirements_latest.txt
```
If PyTorch is already provided by the base image, the build script automatically filters the `torch` wheels when installing the remainder of the stack.

## 🔩 Building CUDA Samples
```bash
cd code
./build_all.sh           # installs deps, builds CUDA samples, runs syntax checks
```
`build_all.sh` iterates over every Makefile, issues `make ARCH=sm_100`, and reports any failures without aborting the whole pass.

## 🧪 Validation & Profiling
| Command | Description |
|---------|-------------|
| `python code/test_blackwell_stack.py` | Sanity suite covering torch.compile, Triton JIT, NVTX, and bandwidth sanity checks. |
| `bash code/profiler_scripts/nsys_profile.sh script.py` | Nsight Systems trace for a workload. |
| `bash code/profiler_scripts/ncu_profile.sh binary` | Nsight Compute metrics for a CUDA sample. |

## 📚 Chapter Layout (Blackwell Focus)
- **Ch1–Ch9**: Foundational CUDA tuning, occupancy, and memory hierarchy – all built for `sm_100`.
- **Ch10–Ch12**: Advanced kernel scheduling, CUDA Graphs, and inter-kernel pipelines with stream-ordered memory.
- **Ch13–Ch16**: PyTorch compilation, Triton kernels, and NVTX profiling pipelines aligned with Blackwell features (TMA, stream-ordered allocators).
- **Ch17–Ch20**: System-level tuning, adaptive token pipelines, and AI-assisted kernel authoring for Blackwell GPUs.

## ✅ What Was Removed
- Automatic architecture detection / switching scripts.
- Blackwell-specific optimisations, banners, and dual-path Makefiles.
- Legacy requirements that referenced CUDA 12.9 or Triton 3.4 pins.

## 📄 Next Steps
1. Ensure CUDA drivers/toolkits for 12.8 are installed on the target Blackwell system.
2. Run `./build_all.sh` to prime the toolchain.
3. Execute chapter-specific workflows (e.g., `python ch16/radix_attention_example.py`).
4. Use the profiler scripts to capture Nsight/System/perf traces as needed.

Welcome to the streamlined, Blackwell-first AI Performance Engineering toolkit! 🖥️⚡
