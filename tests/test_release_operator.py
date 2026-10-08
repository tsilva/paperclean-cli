"""Release operators keep artifact production on the hosted runner."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
SPEC = importlib.util.spec_from_file_location("paperclean_release", SCRIPT)
assert SPEC and SPEC.loader
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def test_local_checks_only_touch_metadata():
    with patch.object(release, "run") as run:
        release.checks()
    assert [call.args[0] for call in run.call_args_list] == [
        ["uv", "lock"],
        ["uv", "lock", "--check"],
        ["git", "diff", "--check"],
    ]


def test_validation_dispatches_exact_remote_commit():
    sha = "a" * 40
    with (
        patch.object(release, "capture", side_effect=["origin/main", sha]),
        patch.object(release, "run") as run,
    ):
        release.validate()
    assert run.call_args_list[-1].args[0] == [
        "gh",
        "workflow",
        "run",
        "release.yml",
        "--ref",
        "main",
        "-f",
        f"ref={sha}",
    ]


def test_validation_rejects_other_upstream():
    with (
        patch.object(release, "capture", return_value="origin/feature"),
        pytest.raises(SystemExit, match="upstream on main"),
    ):
        release.validate()
