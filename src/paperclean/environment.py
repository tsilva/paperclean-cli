"""Safe runtime environment discovery for installed and checkout usage."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from paperclean.errors import ConfigurationError

API_KEY_NAME = "AGENTBRIDGE_API_KEY"
MAX_CONFIG_BYTES = 1_048_576
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_DOTENV_NAMES = frozenset(
    {
        "AGENTBRIDGE_BASE_URL",
        "PAPERCLEAN_AGENTBRIDGE_TIMEOUT",
        "PAPERCLEAN_IMAGE_MODEL",
        "PAPERCLEAN_REVIEW",
        "PAPERCLEAN_REVIEW_MODEL",
        "PAPERCLEAN_MAX_ATTEMPTS",
        "PAPERCLEAN_JOBS",
        "PAPERCLEAN_MAX_COST_USD",
        "PAPERCLEAN_ZDR",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Resolved public configuration and credentials supplied by the process."""

    values: Mapping[str, str]


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
    user_root = config_base.expanduser().resolve() / "paperclean"

    project_values = _dotenv(project_root / ".env")
    user_values = _dotenv(user_root / ".env")

    values = dict(project_values)
    values.update(user_values)
    values.update(process)

    process_key = process.get(API_KEY_NAME, "").strip()
    if process_key:
        values[API_KEY_NAME] = process_key
    else:
        values.pop(API_KEY_NAME, None)
    return RuntimeEnvironment(values)
