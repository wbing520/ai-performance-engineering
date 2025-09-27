# Blackwell Configuration Guide

The repository is now single-architecture: **NVIDIA Blackwell B200/B300 (SM100)**. Blackwell-specific branches, Makefile targets, and docs have been retired to simplify maintenance and keep examples aligned with CUDA 12.9 and PyTorch 2.9 nightlies.

## Key Assets
- `arch_config.py` – returns Blackwell metadata and helper methods.
- `build_all.sh` – installs dependencies, builds CUDA samples with `sm_100`, and runs Python syntax checks.
- `requirements_latest.txt` – canonical dependency list reused by the per-chapter requirement files.

## Usage
```bash
cd code
./build_all.sh                 # install deps + build CUDA samples
python test_blackwell_stack.py        # run sanity suite
bash profiler_scripts/nsys_profile.sh your_script.py  # Nsight Systems trace
```

## Makefile Expectations
Every Makefile (chapters and tools) is normalised to:
- `ARCH ?= sm_100`
- `CUDA_VERSION = 12.9`
- `NVCC_FLAGS` containing `-arch=$(ARCH)`
- Profiling targets: `profile-hta`, `profile-perf`, `profile-all`

Run `bash update_cuda_versions.sh` if you need to reapply the standard flags.

## Dependency Stack
Refer to `requirements_latest.txt` for the full list. Highlights:
- PyTorch 2.9 (nightly, cu129)
- Triton 3.4.0
- CUDA runtime libraries 12.9
- Tooling: Nsight Systems/Compute, psutil, GPUtil, wandb, etc.

## Tips
- If you inherit old branches, run `update_blackwell_requirements.sh` to refresh dependency templates.
- Profiling scripts assume Nsight 2024+ is installed and available on `PATH`.
- For non-Blackwell targets, start a separate fork; this repository remains Blackwell-only by design.
