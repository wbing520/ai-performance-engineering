"""Utility to execute a target script under the PyTorch profiler."""
from __future__ import annotations

import argparse
import importlib.util
import runpy
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a script under torch.profiler")
    parser.add_argument("script", type=Path, help="Path to the Python script to execute")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where profiler artifacts will be written",
    )
    parser.add_argument(
        "--profile-mode",
        default="full",
        choices=["full", "memory", "flops", "modules", "blackwell"],
        help="Profiler preset to use",
    )
    parser.add_argument(
        "--script-args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target script",
    )
    return parser.parse_args()


def _set_env(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("CUDA_CACHE_DISABLE", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")


def _load_module(script: Path) -> ModuleType:
    module_name = script.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def _resolve_entrypoint(module: ModuleType, script: Path) -> Callable[[], None]:
    if hasattr(module, "main") and callable(module.main):  # type: ignore[attr-defined]
        return module.main  # type: ignore[return-value]
    if hasattr(module, "run") and callable(module.run):  # type: ignore[attr-defined]
        return module.run  # type: ignore[return-value]

    def _fallback() -> None:
        runpy.run_path(str(script))

    return _fallback


def _configure_profiler(mode: str) -> dict:
    activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    config = {
        "activities": activities,
        "record_shapes": mode in {"full", "flops", "blackwell"},
        "with_stack": mode in {"full", "blackwell"},
        "with_flops": mode in {"full", "flops", "blackwell"},
        "with_modules": mode in {"full", "modules", "blackwell"},
        "profile_memory": mode in {"full", "memory", "blackwell"},
    }

    # Use immediate collection to support short-running scripts
    config["schedule"] = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
    return config


def _apply_blackwell_tuning() -> None:
    if not torch.cuda.is_available():
        return

    device = torch.cuda.get_device_properties(0)
    compute_capability = f"{device.major}.{device.minor}"

    inductor = getattr(torch, "_inductor", None)
    triton_cfg = getattr(getattr(inductor, "config", None), "triton", None) if inductor else None

    if compute_capability == "10.0" and triton_cfg is not None:
        for attr in (
            "use_blackwell_optimizations",
            "hbm3e_optimizations",
            "tma_support",
            "stream_ordered_memory",
            "nvlink_c2c",
        ):
            if hasattr(triton_cfg, attr):
                setattr(triton_cfg, attr, True)
        if hasattr(triton_cfg, "autotune_mode"):
            triton_cfg.autotune_mode = "max-autotune"

    if triton_cfg is not None and hasattr(triton_cfg, "unique_kernel_names"):
        triton_cfg.unique_kernel_names = True
    dynamo_cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
    if dynamo_cfg is not None and hasattr(dynamo_cfg, "automatic_dynamic_shapes"):
        dynamo_cfg.automatic_dynamic_shapes = True


def run_profiler(script: Path, output_dir: Path, mode: str, script_args: Optional[Iterable[str]]) -> None:
    _set_env(output_dir)
    if mode in {"full", "blackwell"}:
        _apply_blackwell_tuning()

    sys.argv = [str(script)] + list(script_args or [])
    module = _load_module(script)
    entry = _resolve_entrypoint(module, script)

    profiler_kwargs = _configure_profiler(mode)
    start_ts = time.time()
    error: Optional[str] = None

    try:
        with profile(**profiler_kwargs) as prof:
            entry()
            try:
                prof.step()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - surfaced in metadata
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = time.time() - start_ts
        metadata = {
            "script": str(script),
            "args": list(script_args or []),
            "mode": mode,
            "start_time": start_ts,
            "duration_seconds": duration,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "error": error,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        if 'prof' in locals() and getattr(prof, "profiler", None) is not None:
            try:
                trace_path = output_dir / f"chrome_trace_{mode}.json"
                prof.export_chrome_trace(str(trace_path))

                summary_path = output_dir / f"summary_{mode}.txt"
                parts: List[str] = []
                parts.append("Top operators by CUDA time\n")
                try:
                    parts.append(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
                except Exception:
                    parts.append("CUDA metrics unavailable\n")
                parts.append("\nTop operators by CPU time\n")
                parts.append(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))
                if profiler_kwargs.get("profile_memory"):
                    parts.append("\nTop operators by CPU memory usage\n")
                    parts.append(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=25))
                if profiler_kwargs.get("with_flops"):
                    parts.append("\nTop operators by FLOPs\n")
                    try:
                        parts.append(prof.key_averages().table(sort_by="flops", row_limit=25))
                    except Exception:
                        parts.append("FLOP metrics unavailable\n")
                summary_path.write_text("\n".join(parts))
            except AttributeError:
                pass


def main() -> None:
    args = _parse_args()
    run_profiler(args.script, args.output_dir, args.profile_mode, args.script_args)


if __name__ == "__main__":
    main()
