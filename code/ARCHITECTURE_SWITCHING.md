# Blackwell Configuration Guide

The repository is now single-architecture: **NVIDIA Blackwell B200/B300 (SM100)**. Blackwell-specific branches, Makefile targets, and docs have been retired to simplify maintenance and keep examples aligned with CUDA 12.8 and PyTorch 2.8.

## Key Assets
- `arch_config.py` – returns Blackwell metadata and helper methods.
- `build_all.sh` – installs dependencies, builds CUDA samples with `sm_100`, and runs Python syntax checks.
- `requirements_latest.txt` – canonical dependency list reused by the per-chapter requirement files.
- `verify_updates.sh` – lints the tree to ensure no legacy compute-capability 9.x remnants creep back in.

## Usage
```bash
cd code
./build_all.sh                 # install deps + build CUDA samples
python test_architecture_switching.py  # run sanity suite
bash profiler_scripts/nsys_profile.sh your_script.py  # Nsight Systems trace
```

## Makefile Expectations
Every Makefile (chapters and tools) is normalised to:
- `ARCH ?= sm_100`
- `CUDA_VERSION = 12.8`
- `NVCC_FLAGS` containing `-arch=$(ARCH)`
- Profiling targets: `profile-hta`, `profile-perf`, `profile-all`

Run `bash update_cuda_versions.sh` if you need to reapply the standard flags.

## Dependency Stack
Refer to `requirements_latest.txt` for the full list. Highlights:
- PyTorch 2.8 (nightly, cu128)
- Triton 3.3.1
- CUDA runtime libraries 12.8
- Tooling: Nsight Systems/Compute, psutil, GPUtil, wandb, etc.

## Verification Workflow
```bash
cd code
bash verify_updates.sh
```
The script confirms:
- No legacy compute-capability 9.x mentions in active code.
- All Makefiles use the Blackwell toolchain pins.
- Requirements match the Blackwell template (scikit-learn 1.4.2, tokenizers 0.19.1, etc.).

## Tips
- If you inherit old branches, run `update_architecture_switching.sh` to reapply the new requirements template.
- Profiling scripts assume Nsight 2024+ is installed and available on `PATH`.
- For additional architectures, fork this repo—architecture switching has been intentionally removed to keep the mainline clean.
