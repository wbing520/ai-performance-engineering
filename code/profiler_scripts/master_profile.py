#!/usr/bin/env python3
"""Entry point for the master profiling workflow (wrapper around profile_harness)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HARNESS = SCRIPT_DIR / "profile_harness.py"
PYTHON_BIN = os.environ.get("PYTHON", sys.executable)


def _dedupe(items: Iterable[str]) -> List[str]:
    ordered = OrderedDict.fromkeys(items)
    return list(ordered.keys())


def _resolve_target(target: str) -> Path:
    path = Path(target)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _match_example(target: Path) -> Optional[str]:
    from example_registry import EXAMPLES

    resolved = target
    candidates: List[tuple[str, Path]] = []
    for example in EXAMPLES:
        run_path = (REPO_ROOT / example.path).resolve()
        candidates.append((example.name, run_path))
        if example.source is not None:
            source_path = (REPO_ROOT / example.source).resolve()
            candidates.append((example.name, source_path))
    for name, path in candidates:
        if resolved == path:
            return name
    return None


def _merge_examples(arguments: List[str], extra_examples: List[str]) -> List[str]:
    if not extra_examples:
        return list(arguments)

    args = list(arguments)
    try:
        idx = args.index("--examples")
    except ValueError:
        return args + ["--examples", *_dedupe(extra_examples)]

    args.pop(idx)
    existing: List[str] = []
    while idx < len(args) and not args[idx].startswith("--"):
        existing.append(args.pop(idx))

    combined = _dedupe(existing + extra_examples)
    if combined:
        args[idx:idx] = ["--examples", *combined]
    return args


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, remaining = parse_args()

    if args.help:
        return subprocess.call([PYTHON_BIN, str(HARNESS), "--help"] + remaining)

    if args.list and not args.targets:
        return subprocess.call([PYTHON_BIN, str(HARNESS), "--list"] + remaining)

    from example_registry import EXAMPLE_BY_NAME

    resolved_examples: List[str] = []
    for target in args.targets:
        if target in EXAMPLE_BY_NAME:
            resolved_examples.append(target)
            continue

        try:
            path = _resolve_target(target)
        except FileNotFoundError:
            print(f"Unable to resolve path for target: {target}", file=sys.stderr)
            return 1

        match = _match_example(path)
        if match is None:
            print(f"Unable to map target '{target}' to a registered example", file=sys.stderr)
            return 1
        resolved_examples.append(match)

    harness_args = _merge_examples(remaining, resolved_examples)
    command = [PYTHON_BIN, str(HARNESS), *harness_args]
    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
