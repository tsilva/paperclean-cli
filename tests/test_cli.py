from __future__ import annotations

import io
from decimal import Decimal
from pathlib import Path

import pytest

from paperclean.cli import _confirm, _prepare_paths, build_parser, run
from paperclean.errors import ConfigurationError, InputError
from paperclean.preflight import CostProjection, WorkEstimate


def _projection(
    *, account_remaining: Decimal = Decimal("10"), review_enabled: bool = True
) -> CostProjection:
    review_count = 5 if review_enabled else 0
    review_model = "openai/gpt-5.6-sol" if review_enabled else None
    review_provider = "OpenAI" if review_enabled else None
    return CostProjection(
        document_total=1,
        page_total=1,
        max_attempts=3,
        image_model="openai/gpt-image-2",
        image_provider="OpenAI",
        review_enabled=review_enabled,
        review_model=review_model,
        review_provider=review_provider,
        one_pass=WorkEstimate(
            1, review_count, Decimal("1.189452" if review_enabled else "0.344652")
        ),
        configured_max=WorkEstimate(
            3, review_count * 3, Decimal("3.568356" if review_enabled else "1.033956")
        ),
        recovery_ceiling=WorkEstimate(
            6, review_count * 6, Decimal("7.136712" if review_enabled else "2.067912")
        ),
        account_remaining_usd=account_remaining,
        key_remaining_usd=Decimal("10"),
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
    assert "Fidelity reviews" in output
    assert "Paid model calls" in output
    assert "$1.1895" in output
    assert "$10.0000" in output
    assert "READY FOR CONFIRMATION" in output
    assert "╭" in output


def test_balance_covering_one_pass_but_not_recovery_stops_even_with_yes(capsys) -> None:
    with pytest.raises(ConfigurationError, match="recovery-ceiling"):
        _confirm(
            _projection(
                account_remaining=Decimal("0.718589992"),
                review_enabled=False,
            ),
            yes=True,
        )
    output = capsys.readouterr().out
    assert "INSUFFICIENT CREDITS" in output
    assert "$0.7186" in output
    assert "$0.3447" in output
    assert "$2.0679" in output
    assert "required" in output
    assert "recovery" in output
    assert "ceiling" in output
    assert "READY FOR CONFIRMATION" not in output


def test_disabled_review_tui_has_zero_reviews_and_generation_only_cost(capsys) -> None:
    _confirm(_projection(review_enabled=False), yes=True)
    output = capsys.readouterr().out
    assert "Review" in output
    assert "disabled" in output
    assert "SEMANTIC REVIEW DISABLED" in output
    assert "$0.3447" in output


def test_review_boolean_flags_override_in_both_directions() -> None:
    parser = build_parser()
    assert parser.parse_args(["scan.pdf"]).review is None
    assert parser.parse_args(["scan.pdf", "--review"]).review is True
    assert parser.parse_args(["scan.pdf", "--no-review"]).review is False


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
