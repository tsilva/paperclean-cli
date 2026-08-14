#!/usr/bin/env python3
"""Validate, commit, tag, and atomically push a PaperClean release."""

from __future__ import annotations

import argparse
import re
import subprocess
import tomllib
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
HELPER = ROOT / ".codex/skills/build-release/scripts/release_build.py"
VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
DISTRIBUTION = "paperclean-cli"
ARTIFACT_STEM = "paperclean_cli"
TAG_PREFIX = f"{DISTRIBUTION}-v"


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def capture(args: list[str]) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def current_version() -> str:
    with PYPROJECT.open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    if not isinstance(version, str) or VERSION_RE.fullmatch(version) is None:
        raise SystemExit(f"unsupported project version: {version!r}")
    return version


def pypi_unused(version: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/{DISTRIBUTION}/{version}/json", timeout=20
        ):
            return False
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True
        raise


def tag_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", tag],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def patch_bump(version: str) -> str:
    major, minor, patch = (int(item) for item in version.split("."))
    return f"{major}.{minor}.{patch + 1}"


def select_version(requested: str | None) -> str:
    current = current_version()
    version = requested or current
    if requested is None and (tag_exists(f"{TAG_PREFIX}{current}") or not pypi_unused(current)):
        version = patch_bump(current)
    if VERSION_RE.fullmatch(version) is None:
        raise SystemExit("release version must be MAJOR.MINOR.PATCH")
    if not pypi_unused(version):
        raise SystemExit(f"{DISTRIBUTION}=={version} already exists on PyPI")
    if tag_exists(f"{TAG_PREFIX}{version}"):
        raise SystemExit(f"tag {TAG_PREFIX}{version} already exists")
    return version


def ensure_clean_and_synced() -> tuple[str, str]:
    status = capture(["git", "status", "--short"])
    if status:
        raise SystemExit(f"release tree must be clean:\n{status}")
    try:
        upstream = capture(["git", "rev-parse", "--abbrev-ref", "@{u}"])
    except subprocess.CalledProcessError as exc:
        raise SystemExit("current branch must have an upstream") from exc
    remote, separator, branch = upstream.partition("/")
    if not separator:
        raise SystemExit(f"unexpected upstream: {upstream}")
    run(["git", "fetch", "--prune", "--tags", remote])
    ahead, behind = (
        int(item)
        for item in capture(
            ["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"]
        ).split()
    )
    if ahead or behind:
        raise SystemExit(
            f"current branch must be synced with {upstream}; ahead={ahead} behind={behind}"
        )
    return remote, branch


def update_release_files(version: str) -> None:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    pyproject, count = re.subn(
        r'(?m)^version = "[0-9]+\.[0-9]+\.[0-9]+"$',
        f'version = "{version}"',
        pyproject,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not update pyproject version")
    PYPROJECT.write_text(pyproject, encoding="utf-8")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    marker = "## Unreleased\n"
    if marker not in changelog:
        raise SystemExit("CHANGELOG.md must contain an Unreleased section")
    release_heading = re.compile(rf"(?m)^## {re.escape(version)} - \d{{4}}-\d{{2}}-\d{{2}}$")
    if release_heading.search(changelog) is None:
        changelog = changelog.replace(
            marker,
            f"## Unreleased\n\n## {version} - {date.today().isoformat()}\n",
            1,
        )
        CHANGELOG.write_text(changelog, encoding="utf-8")


def checks() -> None:
    commands = [
        ["uv", "lock"],
        ["uv", "lock", "--check"],
        ["uv", "sync", "--frozen", "--all-groups"],
        ["uv", "run", "ruff", "format", "--check", "."],
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "mypy", "-p", "paperclean"],
        ["uv", "run", "pytest"],
        ["uv", "build", "--clear", "--no-sources"],
    ]
    for command in commands:
        run(command)
    distributions = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("*.whl", "*.tar.gz")
        for path in (ROOT / "dist").glob(pattern)
    )
    if len(distributions) != 2:
        raise SystemExit(f"expected one wheel and one sdist, found {distributions}")
    run(["uv", "run", "twine", "check", *distributions])
    run(["uv", "run", "python", str(HELPER), "audit-dist", "--dist-dir", "dist"])
    wheel = next((ROOT / "dist").glob(f"{ARTIFACT_STEM}-*-py3-none-any.whl"))
    run(
        [
            "uv",
            "run",
            "python",
            str(HELPER),
            "smoke-wheel",
            "--wheel",
            str(wheel.relative_to(ROOT)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--to", help="exact MAJOR.MINOR.PATCH version")
    args = parser.parse_args()
    remote, branch = ensure_clean_and_synced()
    version = select_version(args.to)
    if version != current_version():
        update_release_files(version)
    else:
        # Even an already-pending version must gain a dated release heading.
        update_release_files(version)
    checks()
    run(["git", "add", "pyproject.toml", "uv.lock", "CHANGELOG.md"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode != 0:
        run(["git", "commit", "-m", f"Release {DISTRIBUTION} {version}"])
    else:
        print("release-metadata\talready committed")
    tag = f"{TAG_PREFIX}{version}"
    run(["git", "tag", "-a", tag, "-m", f"PaperClean {version}"])
    run(["git", "push", "--atomic", remote, f"HEAD:{branch}", tag])
    print(f"released-tag\t{tag}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
