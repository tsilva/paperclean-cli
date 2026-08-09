#!/usr/bin/env python3
"""Deterministic release artifact and registry checks for PaperClean."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tarfile
import tempfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        value = tomllib.load(stream)["project"]["version"]
    if not isinstance(value, str) or VERSION_RE.fullmatch(value) is None:
        raise SystemExit("project.version must be MAJOR.MINOR.PATCH")
    return value


def check_pypi(version: str) -> None:
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/paperclean/{version}/json", timeout=20
        ) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"unused\tpaperclean=={version}")
            return
        raise
    files = data.get("urls", [])
    if files:
        raise SystemExit(f"paperclean=={version} already exists on PyPI")
    print(f"unused\tpaperclean=={version}")


def wait_pypi(version: str, *, attempts: int = 60) -> None:
    expected = {
        f"paperclean-{version}-py3-none-any.whl",
        f"paperclean-{version}.tar.gz",
    }
    url = f"https://pypi.org/pypi/paperclean/{version}/json"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = json.load(response)
            found = {str(item.get("filename")) for item in data.get("urls", [])}
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            found = set()
        if expected <= found:
            print(f"https://pypi.org/project/paperclean/{version}/")
            for filename in sorted(expected):
                print(filename)
            return
        print(f"waiting for paperclean {version} ({attempt + 1}/{attempts})", flush=True)
        time.sleep(20)
    raise SystemExit(f"paperclean {version} did not appear on PyPI with both files")


def _expected_files(directory: Path) -> tuple[Path, Path]:
    version = project_version()
    wheel = directory / f"paperclean-{version}-py3-none-any.whl"
    sdist = directory / f"paperclean-{version}.tar.gz"
    actual = sorted(path.name for path in directory.glob("paperclean-*") if path.is_file())
    expected = sorted([wheel.name, sdist.name])
    if actual != expected:
        raise SystemExit(f"expected exactly {expected}, found {actual}")
    return wheel, sdist


def audit_dist(directory: Path) -> None:
    wheel, sdist = _expected_files(directory)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if any(name.endswith((".so", ".dylib", ".dll", ".exe")) for name in names):
            raise SystemExit("universal wheel contains a platform binary")
        metadata_name = next((name for name in names if name.endswith(".dist-info/METADATA")), None)
        if metadata_name is None:
            raise SystemExit("wheel has no METADATA")
        metadata = archive.read(metadata_name).decode("utf-8")
        if f"Version: {project_version()}\n" not in metadata:
            raise SystemExit("wheel version does not match pyproject")
        if "paperclean = paperclean.cli:main" not in archive.read(
            metadata_name.replace("METADATA", "entry_points.txt")
        ).decode("utf-8"):
            raise SystemExit("wheel console entry point is missing")
    prefix = f"paperclean-{project_version()}/"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        if not members or any(not member.name.startswith(prefix) for member in members):
            raise SystemExit("sdist has an unexpected root")
        if any(
            part in {".env", ".git", ".venv", "dist"}
            for member in members
            for part in Path(member.name).parts
        ):
            raise SystemExit("sdist contains a forbidden private/build path")
    print(wheel.name)
    print(sdist.name)


def check_tag(tag: str) -> None:
    expected = f"paperclean-v{project_version()}"
    if tag != expected:
        raise SystemExit(f"tag {tag!r} does not match project version; expected {expected!r}")
    print(f"valid\t{tag}")


def smoke_wheel(wheel: Path) -> None:
    expected = f"paperclean {project_version()}"
    with tempfile.TemporaryDirectory(prefix="paperclean-wheel-smoke-") as directory:
        environment = Path(directory) / "venv"
        constraints = Path(directory) / "constraints.txt"
        subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(constraints),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["uv", "venv", str(environment), "--python", "3.11"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        python = environment / "bin" / "python"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--constraint",
                str(constraints),
                str(wheel.resolve()),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result = subprocess.run(
            [str(python), "-m", "paperclean", "--version"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != expected:
            raise SystemExit(
                f"wheel smoke returned {result.stdout.strip()!r}; expected {expected!r}"
            )
    print(f"smoke-ok\t{wheel.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pypi = subparsers.add_parser("check-pypi")
    pypi.add_argument("--version", required=True)
    wait = subparsers.add_parser("wait-pypi")
    wait.add_argument("--version", required=True)
    audit = subparsers.add_parser("audit-dist")
    audit.add_argument("--dist-dir", type=Path, required=True)
    tag = subparsers.add_parser("check-tag")
    tag.add_argument("--tag", required=True)
    smoke = subparsers.add_parser("smoke-wheel")
    smoke.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-pypi":
        check_pypi(args.version)
    elif args.command == "wait-pypi":
        wait_pypi(args.version)
    elif args.command == "audit-dist":
        audit_dist(args.dist_dir.resolve())
    elif args.command == "check-tag":
        check_tag(args.tag)
    elif args.command == "smoke-wheel":
        smoke_wheel(args.wheel)
    else:
        raise AssertionError("unreachable")


if __name__ == "__main__":
    main()
