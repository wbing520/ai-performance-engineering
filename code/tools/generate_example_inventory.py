#!/usr/bin/env python3
"""Generate a manifest of runnable examples grouped by chapter and file type."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_CODE_PREFIX = Path("code")
_ALLOWED_SUFFIXES = {
    ".py": "python",
    ".cu": "cuda",
    ".cpp": "cuda",
    ".cuh": "cuda",
    ".c": "cuda",
    ".cc": "cuda",
    ".sh": "shell",
    ".ipynb": "notebook",
}
_SKIP_DIR_NAMES = {
    "__pycache__",
    "build",
    "profile_runs",
    "profiles",
    ".git",
    ".idea",
    ".vs",
}


@dataclass
class ExampleEntry:
    name: str
    chapter: str
    kind: str
    source: str
    build_command: Optional[str]
    run_command: Optional[str]
    requirements: List[str]
    has_makefile: bool
    readme: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [])}


@dataclass
class ChapterSummary:
    chapter: str
    readme: Optional[str]
    requirements: List[str]
    has_makefile: bool
    example_count: int

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [])}


def _relative_to_code(path: Path) -> Path:
    try:
        return path.relative_to(_CODE_PREFIX)
    except ValueError:
        return path


def _default_build_command(kind: str, rel_source: Path) -> Optional[str]:
    if kind != "cuda":
        return None
    rel_to_code = _relative_to_code(rel_source)
    build_output = Path("build") / rel_to_code.with_suffix("")
    return f"nvcc -O3 -std=c++17 -arch=sm_100 --expt-relaxed-constexpr -o {build_output} {rel_source}"


def _default_run_command(kind: str, rel_source: Path) -> Optional[str]:
    if kind == "python":
        return f"python {rel_source}"
    if kind == "cuda":
        rel_to_code = _relative_to_code(rel_source)
        binary_path = Path("build") / rel_to_code.with_suffix("")
        return str(binary_path)
    if kind == "shell":
        return f"bash {rel_source}"
    if kind == "notebook":
        return f"jupyter nbconvert --execute --to markdown {rel_source}"
    return None


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIR_NAMES for part in path.parts)


def discover_examples(repo_root: Path, include_notebooks: bool = False) -> List[ExampleEntry]:
    code_root = repo_root / _CODE_PREFIX
    if not code_root.exists():
        raise SystemExit(f"Unable to locate code directory at {_CODE_PREFIX}")

    entries: List[ExampleEntry] = []

    for chapter_dir in sorted(code_root.glob("ch*")):
        if not chapter_dir.is_dir():
            continue
        chapter = chapter_dir.name
        has_makefile = (chapter_dir / "Makefile").exists()

        chapter_requirements = sorted(
            str(path.relative_to(repo_root))
            for path in chapter_dir.glob("requirements*.txt")
        )
        readme_path = next((p for p in chapter_dir.glob("README*.md")), None)
        readme_rel = str(readme_path.relative_to(repo_root)) if readme_path else None

        for root, dirnames, filenames in os.walk(chapter_dir):
            root_path = Path(root)
            rel_root = root_path.relative_to(repo_root)
            if _should_skip(rel_root):
                dirnames[:] = []
                continue

            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]

            for filename in sorted(filenames):
                file_path = root_path / filename
                suffix = file_path.suffix.lower()
                if suffix not in _ALLOWED_SUFFIXES:
                    continue
                kind = _ALLOWED_SUFFIXES[suffix]
                if kind == "notebook" and not include_notebooks:
                    continue

                rel_source = file_path.relative_to(repo_root)
                rel_for_name = _relative_to_code(rel_source)
                name = "_".join(rel_for_name.with_suffix("").parts)

                build_command = _default_build_command(kind, rel_source)
                run_command = _default_run_command(kind, rel_source)

                entry = ExampleEntry(
                    name=name,
                    chapter=chapter,
                    kind=kind,
                    source=str(rel_source),
                    build_command=build_command,
                    run_command=run_command,
                    requirements=list(chapter_requirements),
                    has_makefile=has_makefile,
                    readme=readme_rel,
                )
                entries.append(entry)

    entries.sort(key=lambda entry: entry.source)
    return entries


def summarize_by_chapter(entries: Iterable[ExampleEntry]) -> List[ChapterSummary]:
    grouped: Dict[str, List[ExampleEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.chapter, []).append(entry)

    summaries: List[ChapterSummary] = []
    for chapter, items in sorted(grouped.items()):
        readme = next((item.readme for item in items if item.readme), None)
        requirements = sorted({req for item in items for req in item.requirements})
        has_makefile = any(item.has_makefile for item in items)
        summaries.append(
            ChapterSummary(
                chapter=chapter,
                readme=readme,
                requirements=requirements,
                has_makefile=has_makefile,
                example_count=len(items),
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory runnable examples")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON inventory to this path (defaults to stdout)",
    )
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Include Jupyter notebooks in the inventory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    entries = discover_examples(repo_root, include_notebooks=args.include_notebooks)
    summaries = summarize_by_chapter(entries)

    payload = {
        "root": str(repo_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "example_count": len(entries),
        "chapter_count": len(summaries),
        "examples": [entry.to_dict() for entry in entries],
        "chapters": [summary.to_dict() for summary in summaries],
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        json.dump(payload, fp=sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
