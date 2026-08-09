from __future__ import annotations

import io
from pathlib import Path

import pytest

from paperclean.cli import _confirm, _prepare_paths, run
from paperclean.errors import ConfigurationError, InputError


def test_noninteractive_batch_requires_yes(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(ConfigurationError, match="--yes"):
        _confirm(page_total=2, document_total=1, max_attempts=3, yes=False)


def test_output_override_is_not_allowed_for_directory_input(tmp_path: Path) -> None:
    source = tmp_path / "scan.png"
    source.write_bytes(b"not decoded during this check")
    with pytest.raises(InputError, match="one file"):
        _prepare_paths(
            [source],
            output=tmp_path / "result.png",
            force=False,
            directory_input=True,
        )


def test_empty_directory_is_a_fatal_input_error(tmp_path: Path, capsys) -> None:
    assert run([str(tmp_path)]) == 1
    assert "no supported" in capsys.readouterr().err
