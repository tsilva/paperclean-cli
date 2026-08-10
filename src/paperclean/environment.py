"""Safe runtime environment discovery for installed and checkout usage."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from paperclean.errors import ConfigurationError

API_KEY_NAME = "OPENROUTER_API_KEY"
MAX_CONFIG_BYTES = 1_048_576
_BOOTSTRAP_MARKER = "PAPERCLEAN_KEYENV_BOOTSTRAPPED"
_BOOTSTRAP_CWD = "PAPERCLEAN_KEYENV_ORIGINAL_CWD"
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTENV_NAMES = frozenset(
    {
        API_KEY_NAME,
        "OPENROUTER_BASE_URL",
        "PAPERCLEAN_AGENTBRIDGE_BASE_URL",
        "PAPERCLEAN_AGENTBRIDGE_TIMEOUT",
        "PAPERCLEAN_BACKEND",
        "PAPERCLEAN_IMAGE_MODEL",
        "PAPERCLEAN_REVIEW",
        "PAPERCLEAN_REVIEW_MODEL",
        "PAPERCLEAN_MAX_ATTEMPTS",
        "PAPERCLEAN_JOBS",
        "PAPERCLEAN_OCR_LANG",
        "PAPERCLEAN_MAX_COST_USD",
        "PAPERCLEAN_ZDR",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Resolved public configuration plus an optional Keyenv manifest."""

    values: Mapping[str, str]
    keyenv_manifest: Path | None = None


def _project_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    fallback = current
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return fallback
        current = current.parent


def _config_file(path: Path, *, label: str) -> Path | None:
    try:
        if path.is_symlink():
            raise ConfigurationError(f"{label} must not be a symbolic link: {path}")
        if not path.exists():
            return None
        resolved = path.resolve()
        if not resolved.is_file():
            raise ConfigurationError(f"{label} must be a regular file: {path}")
        if resolved.stat().st_size > MAX_CONFIG_BYTES:
            raise ConfigurationError(f"{label} exceeds the 1 MiB size limit: {path}")
        return resolved
    except OSError as exc:
        raise ConfigurationError(f"cannot safely inspect {label}: {path}") from exc


def _dotenv(path: Path) -> dict[str, str]:
    resolved = _config_file(path, label="dotenv file")
    if resolved is None:
        return {}
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"cannot read dotenv file: {resolved}") from exc

    values: dict[str, str] = {}
    for line_number, original in enumerate(lines, start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or _ENV_NAME.fullmatch(name) is None:
            raise ConfigurationError(f"invalid dotenv assignment at {resolved}:{line_number}")
        try:
            tokens = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ConfigurationError(f"invalid dotenv value at {resolved}:{line_number}") from exc
        if name in _DOTENV_NAMES:
            values[name] = " ".join(tokens)
    return values


def _manifest_declaring_api_key(path: Path) -> Path | None:
    resolved = _config_file(path, label="Keyenv manifest")
    if resolved is None:
        return None
    try:
        document = tomllib.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot read Keyenv manifest: {resolved}") from exc
    secrets = document.get("secrets")
    if not isinstance(secrets, dict) or API_KEY_NAME not in secrets:
        return None
    return resolved


def discover_runtime_environment(
    environment: Mapping[str, str] | None = None,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> RuntimeEnvironment:
    """Resolve process, user-level, and repository configuration in priority order."""

    process = dict(os.environ if environment is None else environment)
    working_directory = Path.cwd() if cwd is None else cwd
    project_root = _project_root(working_directory)
    home_directory = Path.home() if home is None else home.expanduser().resolve()
    config_base = Path(process.get("XDG_CONFIG_HOME", home_directory / ".config"))
    user_root = config_base.expanduser().resolve() / "keyenv"

    project_values = _dotenv(project_root / ".env")
    user_values = _dotenv(user_root / ".env")

    values = dict(project_values)
    values.update(user_values)
    values.update(process)

    process_key = process.get(API_KEY_NAME, "").strip()
    user_key = user_values.get(API_KEY_NAME, "").strip()
    project_key = project_values.get(API_KEY_NAME, "").strip()
    if process_key:
        values[API_KEY_NAME] = process_key
        return RuntimeEnvironment(values)
    if user_key:
        values[API_KEY_NAME] = user_key
        return RuntimeEnvironment(values)

    user_manifest = _manifest_declaring_api_key(user_root / ".keyenv.toml")
    if user_manifest is not None:
        values.pop(API_KEY_NAME, None)
        return RuntimeEnvironment(values, user_manifest)
    if project_key:
        values[API_KEY_NAME] = project_key
        return RuntimeEnvironment(values)

    project_manifest = _manifest_declaring_api_key(project_root / ".keyenv.toml")
    if project_manifest is not None:
        values.pop(API_KEY_NAME, None)
        return RuntimeEnvironment(values, project_manifest)
    values.pop(API_KEY_NAME, None)
    return RuntimeEnvironment(values)


def relaunch_with_keyenv(runtime: RuntimeEnvironment, argv: Sequence[str]) -> NoReturn:
    """Replace this process with Keyenv and then a credential-injected PaperClean."""

    manifest = runtime.keyenv_manifest
    if manifest is None:  # pragma: no cover - guarded by the caller
        raise AssertionError("a Keyenv manifest is required")
    keyenv = shutil.which("keyenv", path=runtime.values.get("PATH"))
    if keyenv is None:
        raise ConfigurationError(
            f"{manifest} requires the `keyenv` command; install `keyenv-macos`"
        )

    original_cwd = Path.cwd().resolve()
    child_environment = dict(runtime.values)
    child_environment[_BOOTSTRAP_MARKER] = "1"
    child_environment[_BOOTSTRAP_CWD] = os.fspath(original_cwd)
    command = [
        keyenv,
        "run",
        "--manifest",
        os.fspath(manifest),
        "--",
        sys.executable,
        "-m",
        "paperclean",
        *argv,
    ]
    try:
        os.chdir(manifest.parent)
        os.execvpe(keyenv, command, child_environment)
    except OSError as exc:
        os.chdir(original_cwd)
        raise ConfigurationError(f"could not launch `keyenv` for {manifest}") from exc
    raise AssertionError("os.execvpe unexpectedly returned")  # pragma: no cover


def restore_keyenv_working_directory() -> None:
    """Restore the caller's directory after a global-manifest Keyenv relaunch."""

    marker = os.environ.pop(_BOOTSTRAP_MARKER, None)
    original = os.environ.pop(_BOOTSTRAP_CWD, None)
    if marker is None and original is None:
        return
    if marker != "1" or not original:
        raise ConfigurationError("invalid Keyenv bootstrap state")
    target = Path(original)
    try:
        resolved = target.resolve(strict=True)
        if not resolved.is_dir():
            raise ConfigurationError("original Keyenv working directory is not a directory")
        os.chdir(resolved)
    except OSError as exc:
        raise ConfigurationError("cannot restore the original Keyenv working directory") from exc
