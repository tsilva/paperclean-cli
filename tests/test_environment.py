from __future__ import annotations

from pathlib import Path

import pytest

from paperclean.environment import API_KEY_NAME, discover_runtime_environment
from paperclean.errors import ConfigurationError


def configs(tmp_path: Path) -> tuple[Path, Path, Path]:
    home = tmp_path / "home"
    user = home / ".config" / "paperclean"
    user.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    return home, user, project


def test_user_nonsecret_settings_override_repository_settings(tmp_path: Path) -> None:
    home, user, project = configs(tmp_path)
    (project / ".env").write_text("PAPERCLEAN_JOBS=2\nAGENTBRIDGE_API_KEY=stale-project\n")
    (user / ".env").write_text(
        "PAPERCLEAN_JOBS=4\nPAPERCLEAN_REVIEW=true\nAGENTBRIDGE_API_KEY=stale-user\n"
    )
    runtime = discover_runtime_environment({}, cwd=project, home=home)
    assert runtime.values["PAPERCLEAN_JOBS"] == "4"
    assert runtime.values["PAPERCLEAN_REVIEW"] == "true"
    assert API_KEY_NAME not in runtime.values


def test_process_credentials_and_settings_override_local_files(tmp_path: Path) -> None:
    home, user, project = configs(tmp_path)
    (user / ".env").write_text("AGENTBRIDGE_API_KEY=stale-user\nPAPERCLEAN_JOBS=4\n")
    runtime = discover_runtime_environment(
        {API_KEY_NAME: "dummy-process", "PAPERCLEAN_JOBS": "6"}, cwd=project, home=home
    )
    assert runtime.values[API_KEY_NAME] == "dummy-process"
    assert runtime.values["PAPERCLEAN_JOBS"] == "6"


def test_legacy_manifests_and_global_dotenv_cannot_supply_credentials(tmp_path: Path) -> None:
    home, _, project = configs(tmp_path)
    legacy = home / ".config" / "keyenv"
    legacy.mkdir(parents=True)
    (legacy / ".env").write_text("AGENTBRIDGE_API_KEY=stale-global\nPAPERCLEAN_JOBS=99\n")
    (legacy / ".keyenv.toml").write_text("invalid legacy manifest")
    (project / ".keyenv.toml").write_text("invalid legacy manifest")
    runtime = discover_runtime_environment({}, cwd=project, home=home)
    assert API_KEY_NAME not in runtime.values
    assert "PAPERCLEAN_JOBS" not in runtime.values


def test_empty_process_credential_cannot_fall_back_to_dotenv(tmp_path: Path) -> None:
    home, user, project = configs(tmp_path)
    (user / ".env").write_text("AGENTBRIDGE_API_KEY=stale-user\n")
    runtime = discover_runtime_environment({API_KEY_NAME: " "}, cwd=project, home=home)
    assert API_KEY_NAME not in runtime.values


def test_dotenv_parse_error_does_not_echo_value(tmp_path: Path) -> None:
    home, user, project = configs(tmp_path)
    value = "dummy-do-not-print"
    (user / ".env").write_text(f'PAPERCLEAN_IMAGE_MODEL="{value}\n')
    with pytest.raises(ConfigurationError) as raised:
        discover_runtime_environment({}, cwd=project, home=home)
    assert value not in str(raised.value)
