from __future__ import annotations

import os
from pathlib import Path

import pytest

from paperclean.environment import (
    API_KEY_NAME,
    discover_runtime_environment,
    restore_keyenv_working_directory,
)
from paperclean.errors import ConfigurationError


def _manifest(path: Path) -> None:
    path.write_text(
        """[keyenv]
version = 1

[secrets.OPENROUTER_API_KEY]
account = "test/OPENROUTER_API_KEY"
required = true
""",
        encoding="utf-8",
    )


def test_user_dotenv_overrides_repository_dotenv(tmp_path: Path) -> None:
    home = tmp_path / "home"
    user_config = home / ".config" / "keyenv"
    user_config.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    (project / ".env").write_text(
        "OPENROUTER_API_KEY=repository\nPAPERCLEAN_JOBS=2\n", encoding="utf-8"
    )
    (user_config / ".env").write_text(
        "OPENROUTER_API_KEY='user secret'\nPAPERCLEAN_JOBS=4\nPAPERCLEAN_REVIEW=true\n",
        encoding="utf-8",
    )

    runtime = discover_runtime_environment({}, cwd=project, home=home)

    assert runtime.values[API_KEY_NAME] == "user secret"
    assert runtime.values["PAPERCLEAN_JOBS"] == "4"
    assert runtime.values["PAPERCLEAN_REVIEW"] == "true"
    assert runtime.keyenv_manifest is None


def test_process_environment_has_highest_priority(tmp_path: Path) -> None:
    home = tmp_path / "home"
    user_config = home / ".config" / "keyenv"
    user_config.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    (user_config / ".env").write_text("OPENROUTER_API_KEY=user\n", encoding="utf-8")

    runtime = discover_runtime_environment({API_KEY_NAME: "process"}, cwd=project, home=home)

    assert runtime.values[API_KEY_NAME] == "process"
    assert runtime.keyenv_manifest is None


def test_user_manifest_precedes_repository_dotenv(tmp_path: Path) -> None:
    home = tmp_path / "home"
    user_config = home / ".config" / "keyenv"
    user_config.mkdir(parents=True)
    _manifest(user_config / ".keyenv.toml")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    (project / ".env").write_text("OPENROUTER_API_KEY=repository\n", encoding="utf-8")

    runtime = discover_runtime_environment({}, cwd=project, home=home)

    assert API_KEY_NAME not in runtime.values
    assert runtime.keyenv_manifest == (user_config / ".keyenv.toml").resolve()


def test_repository_manifest_is_the_final_fallback(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    _manifest(project / ".keyenv.toml")

    runtime = discover_runtime_environment({}, cwd=project, home=home)

    assert runtime.keyenv_manifest == (project / ".keyenv.toml").resolve()


def test_dotenv_parse_error_does_not_echo_secret(tmp_path: Path) -> None:
    home = tmp_path / "home"
    config = home / ".config" / "keyenv"
    config.mkdir(parents=True)
    secret = "do-not-print-this-value"
    (config / ".env").write_text(f'OPENROUTER_API_KEY="{secret}\n', encoding="utf-8")

    with pytest.raises(ConfigurationError) as raised:
        discover_runtime_environment({}, cwd=tmp_path, home=home)

    assert secret not in str(raised.value)


def test_restore_keyenv_working_directory(tmp_path: Path, monkeypatch) -> None:
    original = tmp_path / "original"
    bootstrap = tmp_path / "bootstrap"
    original.mkdir()
    bootstrap.mkdir()
    monkeypatch.chdir(bootstrap)
    monkeypatch.setenv("PAPERCLEAN_KEYENV_BOOTSTRAPPED", "1")
    monkeypatch.setenv("PAPERCLEAN_KEYENV_ORIGINAL_CWD", os.fspath(original))

    restore_keyenv_working_directory()

    assert Path.cwd() == original.resolve()
    assert "PAPERCLEAN_KEYENV_BOOTSTRAPPED" not in os.environ
    assert "PAPERCLEAN_KEYENV_ORIGINAL_CWD" not in os.environ
