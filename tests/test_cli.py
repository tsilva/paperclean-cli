from __future__ import annotations

import io
from decimal import Decimal
from pathlib import Path

import pytest

from paperclean.cli import _confirm, _prepare_paths, build_parser, run
from paperclean.errors import ConfigurationError, InputError
from paperclean.preflight import CostProjection, WorkEstimate, build_subscription_projection


def _projection(*, account_remaining: Decimal = Decimal("20")) -> CostProjection:
    review_count = 5
    return CostProjection(
        document_total=1,
        page_total=1,
        max_attempts=3,
        image_model="openai/gpt-image-2",
        image_provider="OpenAI",
        review_model="openai/gpt-5.6-sol",
        review_provider="OpenAI",
        one_pass=WorkEstimate(1, review_count, Decimal("1.189452")),
        configured_max=WorkEstimate(3, review_count * 3, Decimal("3.568356")),
        recovery_ceiling=WorkEstimate(9, review_count * 12, Decimal("13.239468")),
        account_remaining_usd=account_remaining,
        key_remaining_usd=Decimal("20"),
        key_unlimited=False,
        soft_limit_usd=None,
    )


def test_noninteractive_single_page_requires_yes(monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(ConfigurationError, match="--yes"):
        _confirm(_projection(), yes=False)


def test_preflight_prints_work_costs_and_low_balance_warning(capsys) -> None:
    _confirm(_projection(), yes=True)
    output = capsys.readouterr().out
    assert "Work" in output
    assert "One pass" in output
    assert "Configured max" in output
    assert "Recovery ceiling" in output
    assert "Image generations" in output
    assert "Fidelity verifications" in output
    assert "Paid model calls" in output
    assert "$1.1895" in output
    assert "$20.0000" in output
    assert "READY FOR CONFIRMATION" in output
    assert "╭" in output


def test_balance_covering_one_pass_but_not_recovery_stops_even_with_yes(capsys) -> None:
    with pytest.raises(ConfigurationError, match="recovery-ceiling"):
        _confirm(
            _projection(account_remaining=Decimal("2")),
            yes=True,
        )
    output = capsys.readouterr().out
    assert "INSUFFICIENT CREDITS" in output
    assert "$2.0000" in output
    assert "$1.1895" in output
    assert "$13.2395" in output
    assert "required" in output
    assert "recovery" in output
    assert "ceiling" in output
    assert "READY FOR CONFIRMATION" not in output


def test_codex_subscription_preflight_reports_calls_without_inventing_usd(capsys) -> None:
    projection = build_subscription_projection(
        document_total=1,
        page_total=2,
        max_attempts=3,
        image_model="codex/gpt-5.6-sol",
        review_model="codex/gpt-5.6-sol",
        backend_version="0.1.9",
    )

    _confirm(projection, yes=True)

    output = capsys.readouterr().out
    assert "Codex-work preflight" in output
    assert "Codex subscription" in output
    assert "USD cost" in output
    assert "unavailable" in output
    assert "Model calls" in output
    assert "Paid model calls" not in output


def test_removed_review_and_ocr_flags_are_rejected() -> None:
    parser = build_parser()
    parser.parse_args(["scan.pdf"])
    with pytest.raises(SystemExit):
        parser.parse_args(["scan.pdf", "--no-review"])
    with pytest.raises(SystemExit):
        parser.parse_args(["scan.pdf", "--ocr-lang", "eng"])


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
