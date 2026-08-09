"""Input enumeration and collision-free output naming."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from paperclean.errors import InputError, OutputCollisionError

SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True, slots=True)
class OutputPaths:
    source: Path
    output: Path
    report: Path


def is_generated_path(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.stem.lower().endswith(".clean")
        or name.endswith(".report.json")
        or name.startswith(".paperclean-")
    )


def is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def discover(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        if not is_supported(path):
            raise InputError(f"unsupported input type: {path}")
        if is_generated_path(path):
            raise InputError(f"refusing to clean an existing PaperClean output: {path}")
        return [path]
    if not path.is_dir():
        raise InputError(f"input does not exist: {path}")

    found: list[Path] = []
    for root, directories, filenames in os.walk(path, followlinks=False):
        directories[:] = sorted(
            directory for directory in directories if not (Path(root) / directory).is_symlink()
        )
        for filename in sorted(filenames):
            candidate = Path(root) / filename
            if candidate.is_symlink() or is_generated_path(candidate):
                continue
            if is_supported(candidate):
                found.append(candidate.resolve())
    return found


def output_paths(source: Path, override: Path | None = None) -> OutputPaths:
    source = source.resolve()
    if override is None:
        output = source.with_name(f"{source.stem}.clean{source.suffix}")
    else:
        output = override.expanduser().resolve()
        source_type = (
            "jpeg" if source.suffix.lower() in {".jpg", ".jpeg"} else source.suffix.lower()
        )
        output_type = (
            "jpeg" if output.suffix.lower() in {".jpg", ".jpeg"} else output.suffix.lower()
        )
        if output.suffix.lower() not in SUPPORTED_SUFFIXES or source_type != output_type:
            raise InputError("--output must have a compatible PDF or image suffix")
        if output == source:
            raise InputError("--output cannot overwrite the source path")
    report = output.with_name(f"{output.name}.report.json")
    return OutputPaths(source=source, output=output, report=report)


def check_collision(paths: OutputPaths, *, force: bool) -> None:
    existing = [path for path in (paths.output, paths.report) if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise OutputCollisionError(f"output collision: {joined}; use --force to replace")
