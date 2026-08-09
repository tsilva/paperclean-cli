"""Filesystem and hashing helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def private_write(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def staged_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".paperclean-{target.name}-", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    os.chmod(name, 0o600)
    return Path(name)


@contextmanager
def private_workdir() -> Iterator[Path]:
    directory = Path(tempfile.mkdtemp(prefix="paperclean-"))
    os.chmod(directory, 0o700)
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def publish_pair(
    staged_output: Path,
    staged_report: Path,
    output: Path,
    report: Path,
    *,
    force: bool,
) -> None:
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        if force:
            for target in (output, report):
                if target.exists():
                    backup = target.with_name(f".paperclean-{target.name}-{uuid4().hex}.bak")
                    os.replace(target, backup)
                    backups[target] = backup
        os.replace(staged_output, output)
        replaced.append(output)
        os.replace(staged_report, report)
        replaced.append(report)
    except BaseException:
        for target in reversed(replaced):
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
