"""Unified harness for running Nsight Systems, Nsight Compute, and torch.profiler."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from example_registry import EXAMPLE_BY_NAME, EXAMPLES, Example

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
DEFAULT_TIMEOUT = 900  # seconds


@dataclass
class RunResult:
    profiler: str
    example: Example
    command: List[str]
    output_dir: Path
    stdout_path: Path
    stderr_path: Path
    duration: float
    exit_code: int
    skipped: bool
    skip_reason: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profiling harness for all chapter examples")
    parser.add_argument(
        "--examples",
        nargs="*",
        default=["all"],
        help="Example names to run (default: all). Use --list to view names",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Only run examples matching these tags",
    )
    parser.add_argument(
        "--profile",
        nargs="*",
        default=["all"],
        choices=["all", "nsys", "ncu", "pytorch"],
        help="Profilers to run (default: all)",
    )
    parser.add_argument(
        "--profile-mode",
        action="append",
        choices=["full", "memory", "flops", "modules", "blackwell"],
        help="PyTorch profiler mode (repeat to collect multiple modes)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "profiles",
        help="Root directory for profiler outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs when the target output directory already exists",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Limit the number of examples executed",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available examples and exit",
    )
    return parser.parse_args()


def format_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)



def missing_modules(modules: Iterable[str]) -> List[str]:
    import importlib.util

    missing: List[str] = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def select_examples(names: List[str], tags: Optional[Iterable[str]]) -> List[Example]:
    if "all" in names:
        selected = list(EXAMPLES)
    else:
        unknown = [name for name in names if name not in EXAMPLE_BY_NAME]
        if unknown:
            raise SystemExit(f"Unknown example name(s): {', '.join(sorted(unknown))}")
        selected = [EXAMPLE_BY_NAME[name] for name in names]

    if tags:
        tags_set = set(tags)
        selected = [ex for ex in selected if tags_set.intersection(ex.tags)]

    return selected


def resolve_profilers(requested: List[str]) -> List[str]:
    if "all" in requested:
        return ["nsys", "ncu", "pytorch"]
    dedup: List[str] = []
    for item in requested:
        if item not in dedup and item != "all":
            dedup.append(item)
    return dedup


def session_directory(root: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    directory = root / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_command(
    command: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
    dry_run: bool,
) -> Tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] {format_command(command)}")
        return 0, 0.0

    start = time.time()
    with stdout_path.open("w") as stdout_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
        )
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            exit_code = -1
    duration = time.time() - start
    return exit_code, duration


def base_env(example: Example) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(example.env)

    chapter_tags = {
        "ch01",
        "ch02",
        "ch13",
        "ch14",
        "ch15",
        "ch16",
        "ch17",
        "ch18",
        "ch19",
        "ch20",
    }
    if chapter_tags.intersection(example.tags):
        env.setdefault("TORCHINDUCTOR_AUTOTUNE", "0")
        env.setdefault("TORCH_COMPILE_DISABLE", "1")
    return env


def _terminate_lingering_nsys() -> None:
    """Best-effort cleanup for stray Nsight Systems agents before NCU runs."""
    try:
        subprocess.run(["pkill", "-f", "nsys"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def check_preconditions(example: Example, profiler: str) -> Optional[str]:
    import shutil

    missing = missing_modules(example.requires_modules)
    if missing:
        return f"missing modules: {', '.join(missing)}"

    command_requirements = list(example.requires_commands)
    if profiler == "nsys":
        command_requirements.append("nsys")
    elif profiler == "ncu":
        command_requirements.append("ncu")

    unavailable = [cmd for cmd in command_requirements if shutil.which(cmd) is None]
    if unavailable:
        return f"missing commands: {', '.join(sorted(set(unavailable)))}"

    if example.requires_cuda and not torch.cuda.is_available():
        if example.allow_cpu_fallback:
            return "CUDA device unavailable"
        return "CUDA device required"

    if example.min_cuda_gpus:
        if not torch.cuda.is_available() or torch.cuda.device_count() < example.min_cuda_gpus:
            return f"needs >= {example.min_cuda_gpus} CUDA device(s)"

    return None


def profiler_output_dir(session_dir: Path, profiler: str, example: Example) -> Path:
    return session_dir / profiler / example.name


def run_nsys(example: Example, session_dir: Path, context: argparse.Namespace, timeout: int) -> RunResult:
    out_dir = profiler_output_dir(session_dir, "nsys", example)
    out_base = out_dir / f"nsys_{example.name}"
    command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "-o",
        str(out_base),
        "-t",
        "cuda,nvtx,osrt,cudnn,cublas",
        "-s",
        "cpu",
        "--python-sampling=true",
        "--python-sampling-frequency=1000",
        "--cudabacktrace=true",
        "--stats=true",
        "python",
        str(example.resolved_path(REPO_ROOT)),
        *example.default_args,
    ]

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    exit_code, duration = run_command(
        command,
        cwd=REPO_ROOT,
        env=base_env(example),
        timeout=timeout,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        dry_run=context.dry_run,
    )
    (out_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

    return RunResult(
        profiler="nsys",
        example=example,
        command=command,
        output_dir=out_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration=duration,
        exit_code=exit_code,
        skipped=context.dry_run,
        skip_reason="dry-run" if context.dry_run else None,
    )


def run_ncu(example: Example, session_dir: Path, context: argparse.Namespace, timeout: int) -> RunResult:
    out_dir = profiler_output_dir(session_dir, "ncu", example)
    out_base = out_dir / f"ncu_{example.name}"
    _terminate_lingering_nsys()
    command = [
        "ncu",
        "--set",
        "full",
        "-o",
        str(out_base),
        "python",
        str(example.resolved_path(REPO_ROOT)),
        *example.default_args,
    ]

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"

    exit_code, duration = run_command(
        command,
        cwd=REPO_ROOT,
        env=base_env(example),
        timeout=timeout,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        dry_run=context.dry_run,
    )
    (out_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

    return RunResult(
        profiler="ncu",
        example=example,
        command=command,
        output_dir=out_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration=duration,
        exit_code=exit_code,
        skipped=context.dry_run,
        skip_reason="dry-run" if context.dry_run else None,
    )


def run_pytorch_profiler(
    example: Example,
    session_dir: Path,
    context: argparse.Namespace,
    modes: Sequence[str],
    timeout: int,
) -> List[RunResult]:
    results: List[RunResult] = []
    runner = Path(__file__).resolve().with_name("pytorch_profiler_runner.py")

    for mode in modes:
        out_dir = profiler_output_dir(session_dir, f"pytorch_{mode}", example)
        command = [
            PYTHON,
            str(runner),
            str(example.resolved_path(REPO_ROOT)),
            "--output-dir",
            str(out_dir),
            "--profile-mode",
            mode,
        ]
        if example.default_args:
            command.append("--script-args")
            command.extend(example.default_args)

        stdout_path = out_dir / "stdout.log"
        stderr_path = out_dir / "stderr.log"

        exit_code, duration = run_command(
            command,
            cwd=REPO_ROOT,
            env=base_env(example),
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            dry_run=context.dry_run,
        )
        (out_dir / "command.json").write_text(json.dumps({"command": command}, indent=2))

        results.append(
            RunResult(
                profiler=f"pytorch_{mode}",
                example=example,
                command=command,
                output_dir=out_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                duration=duration,
                exit_code=exit_code,
                skipped=context.dry_run,
                skip_reason="dry-run" if context.dry_run else None,
            )
        )
    return results


def summarize(results: List[RunResult], session_dir: Path) -> None:
    summary = []
    for result in results:
        summary.append(
            {
                "profiler": result.profiler,
                "example": result.example.name,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration,
                "command": result.command,
                "stdout": str(result.stdout_path),
                "stderr": str(result.stderr_path),
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
            }
        )
    (session_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def maybe_skip_output(out_dir: Path, skip_existing: bool) -> bool:
    return skip_existing and out_dir.exists()


def main() -> None:
    args = parse_args()

    if args.list:
        for example in EXAMPLES:
            print(f"{example.name:35s} :: tags={','.join(example.tags)} :: path={example.path}")
        return

    profilers = resolve_profilers(args.profile)
    pytorch_modes = args.profile_mode or ["full"]

    selected = select_examples(args.examples, args.tags)
    if args.max_examples is not None:
        selected = selected[: args.max_examples]

    session_dir = session_directory(args.output_root)
    all_results: List[RunResult] = []

    import torch  # local import so command listing works without CUDA

    for example in selected:
        timeout = example.timeout_seconds or DEFAULT_TIMEOUT
        for profiler in profilers:
            if profiler == "pytorch":
                modes_to_run: List[str] = []
                for mode in pytorch_modes:
                    out_dir = profiler_output_dir(session_dir, f"pytorch_{mode}", example)
                    if maybe_skip_output(out_dir, args.skip_existing):
                        all_results.append(
                            RunResult(
                                profiler=f"pytorch_{mode}",
                                example=example,
                                command=[],
                                output_dir=out_dir,
                                stdout_path=out_dir / "stdout.log",
                                stderr_path=out_dir / "stderr.log",
                                duration=0.0,
                                exit_code=0,
                                skipped=True,
                                skip_reason="existing-output",
                            )
                        )
                    else:
                        modes_to_run.append(mode)
                if not modes_to_run:
                    continue

                reason = check_preconditions(example, "pytorch")
                if reason:
                    print(f"[skip] {example.name} (pytorch) -> {reason}")
                    for mode in modes_to_run:
                        out_dir = profiler_output_dir(session_dir, f"pytorch_{mode}", example)
                        all_results.append(
                            RunResult(
                                profiler=f"pytorch_{mode}",
                                example=example,
                                command=[],
                                output_dir=out_dir,
                                stdout_path=out_dir / "stdout.log",
                                stderr_path=out_dir / "stderr.log",
                                duration=0.0,
                                exit_code=0,
                                skipped=True,
                                skip_reason=reason,
                            )
                        )
                    continue

                results = run_pytorch_profiler(example, session_dir, args, modes_to_run, timeout)
                all_results.extend(results)
                continue

            out_dir = profiler_output_dir(session_dir, profiler, example)
            if maybe_skip_output(out_dir, args.skip_existing):
                all_results.append(
                    RunResult(
                        profiler=profiler,
                        example=example,
                        command=[],
                        output_dir=out_dir,
                        stdout_path=out_dir / "stdout.log",
                        stderr_path=out_dir / "stderr.log",
                        duration=0.0,
                        exit_code=0,
                        skipped=True,
                        skip_reason="existing-output",
                    )
                )
                continue

            reason = check_preconditions(example, profiler)
            if reason:
                print(f"[skip] {example.name} ({profiler}) -> {reason}")
                all_results.append(
                    RunResult(
                        profiler=profiler,
                        example=example,
                        command=[],
                        output_dir=out_dir,
                        stdout_path=out_dir / "stdout.log",
                        stderr_path=out_dir / "stderr.log",
                        duration=0.0,
                        exit_code=0,
                        skipped=True,
                        skip_reason=reason,
                    )
                )
                continue

            if profiler == "nsys":
                result = run_nsys(example, session_dir, args, timeout)
                all_results.append(result)
            elif profiler == "ncu":
                result = run_ncu(example, session_dir, args, timeout)
                all_results.append(result)
            else:
                raise AssertionError(f"Unknown profiler {profiler}")

    summarize(all_results, session_dir)

    failed = [r for r in all_results if not r.skipped and r.exit_code != 0]
    if failed:
        print("\nFailures detected:")
        for item in failed:
            print(f" - {item.example.name} [{item.profiler}] (exit={item.exit_code})")
        sys.exit(1)


if __name__ == "__main__":
    import shutil
    import torch

    main()
