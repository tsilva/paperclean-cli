from __future__ import annotations

from pathlib import Path

import pytest

from paperclean.config import Settings
from paperclean.discovery import discover, output_paths
from paperclean.errors import ConfigurationError, InputError


def test_settings_precedence_and_cost_serialization() -> None:
    settings = Settings.from_sources(
        {"jobs": 4, "image_model": "vendor/cli-model"},
        {
            "OPENROUTER_API_KEY": "secret",
            "PAPERCLEAN_IMAGE_MODEL": "vendor/env-model",
            "PAPERCLEAN_REVIEW_MODEL": "vendor/reviewer",
            "PAPERCLEAN_MAX_COST_USD": "1.25",
        },
    )
    assert settings.image_model == "openrouter/vendor/cli-model"
    assert settings.review_model == "openrouter/vendor/reviewer"
    assert settings.jobs == 4
    assert settings.paid_jobs == 1


def test_legacy_review_toggle_is_rejected_because_verification_is_mandatory() -> None:
    with pytest.raises(ConfigurationError, match="always enabled"):
        Settings.from_sources(
            {},
            {"OPENROUTER_API_KEY": "secret", "PAPERCLEAN_REVIEW": "false"},
        )


def test_settings_ignore_upstream_secret_and_need_no_gateway_token() -> None:
    assert Settings.from_sources({}, {"OPENROUTER_API_KEY": "must-not-forward"}).api_key == ""


def test_agentbridge_settings_need_no_openrouter_key_and_use_codex_defaults() -> None:
    settings = Settings.from_sources(
        {"image_model": "codex/gpt-6.1-sol", "review_model": "codex/gpt-6.1-sol"}, {}
    )

    assert settings.api_key == ""
    assert settings.base_url == "http://127.0.0.1:8082/api/v1"
    assert settings.image_model == "codex/gpt-6.1-sol"
    assert settings.review_model == "codex/gpt-6.1-sol"
    assert settings.agentbridge_timeout == 660


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"backend": "agentbridge", "base_url": "https://openrouter.ai/api/v1"},
            "AgentBridge endpoint",
        ),
        (
            {
                "image_model": "codex/gpt-6.1-sol",
                "review_model": "codex/gpt-6.1-sol",
                "max_cost_usd": "1",
            },
            "max-cost-usd",
        ),
        (
            {"image_model": "codex/gpt-6.1-sol", "review_model": "codex/gpt-6.1-sol", "zdr": True},
            "zdr",
        ),
        (
            {"image_model": "openai/gpt-image-2", "review_model": "codex/gpt-6.1-sol"},
            "same model family",
        ),
    ],
)
def test_agentbridge_settings_reject_incompatible_options(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        Settings.from_sources(overrides, {})


def test_discover_recursive_skips_outputs_and_symlinks(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "page.PNG"
    source.write_bytes(b"x")
    (nested / "page.clean.PNG").write_bytes(b"x")
    (nested / "page.clean.PNG.report.json").write_bytes(b"x")
    (tmp_path / "link.png").symlink_to(source)
    loop = nested / "loop"
    loop.symlink_to(tmp_path, target_is_directory=True)
    assert discover(tmp_path) == [source.resolve()]


def test_output_override_preserves_document_family(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"x")
    assert output_paths(source).output.name == "scan.clean.pdf"
    with pytest.raises(InputError, match="compatible"):
        output_paths(source, tmp_path / "scan.png")
