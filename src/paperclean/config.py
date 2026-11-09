"""Environment and CLI configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from urllib.parse import urlparse

from paperclean.errors import ConfigurationError

DEFAULT_BASE_URL = "http://127.0.0.1:8082/api/v1"
DEFAULT_IMAGE_MODEL = "openai/gpt-image-2"
DEFAULT_REVIEW_MODEL = "openai/gpt-5.6-sol"
DEFAULT_AGENTBRIDGE_BASE_URL = "http://127.0.0.1:8082/api/v1"
DEFAULT_AGENTBRIDGE_MODEL = "codex/gpt-6.1-sol"
DEFAULT_AGENTBRIDGE_TIMEOUT = 660
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_JOBS = 1


def _positive_int(name: str, value: object) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return parsed


def _optional_money(name: str, value: object | None) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a positive decimal amount") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be a positive decimal amount")
    return parsed


def _bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str
    backend: Literal["agentbridge"] = "agentbridge"
    base_url: str = DEFAULT_BASE_URL
    image_model: str = DEFAULT_IMAGE_MODEL
    review_model: str = DEFAULT_REVIEW_MODEL
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    jobs: int = DEFAULT_JOBS
    max_cost_usd: Decimal | None = None
    zdr: bool = False
    render_dpi: int = 300
    max_reference_edge: int = 4096
    min_effective_dpi: int = 150
    agentbridge_timeout: int = DEFAULT_AGENTBRIDGE_TIMEOUT

    @property
    def paid_jobs(self) -> int:
        return 1 if self.max_cost_usd is not None else self.jobs

    @classmethod
    def from_sources(
        cls,
        overrides: Mapping[str, Any],
        environ: Mapping[str, str] | None = None,
    ) -> Settings:
        env = os.environ if environ is None else environ

        def choose(key: str, env_key: str, default: object) -> object:
            value = overrides.get(key)
            if value is not None:
                return value
            return env.get(env_key, default)

        api_key = str(choose("api_key", "AGENTBRIDGE_API_KEY", "")).strip()
        base_url = str(choose("base_url", "AGENTBRIDGE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        url_name = "AGENTBRIDGE_BASE_URL"
        if not base_url.startswith(("https://", "http://")):
            raise ConfigurationError(f"{url_name} must be an HTTP(S) URL")
        parsed_base = urlparse(base_url)
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        if not parsed_base.hostname or parsed_base.hostname == "openrouter.ai":
            raise ConfigurationError("Configure an AgentBridge endpoint")
        if parsed_base.scheme == "http" and parsed_base.hostname not in loopback_hosts:
            raise ConfigurationError(f"{url_name} must use HTTPS unless it targets localhost")
        default_image_model = DEFAULT_IMAGE_MODEL
        default_review_model = DEFAULT_REVIEW_MODEL
        image_model = str(
            choose("image_model", "PAPERCLEAN_IMAGE_MODEL", default_image_model)
        ).strip()
        review_model = str(
            choose("review_model", "PAPERCLEAN_REVIEW_MODEL", default_review_model)
        ).strip()
        if "PAPERCLEAN_REVIEW" in env:
            raise ConfigurationError(
                "PAPERCLEAN_REVIEW was removed; model verification is always enabled"
            )
        if "/" not in image_model or "/" not in review_model:
            raise ConfigurationError("model identifiers must use author/model form")
        image_model = gateway_model(image_model)
        review_model = gateway_model(review_model)
        native_codex = image_model.startswith("codex/")
        if native_codex != review_model.startswith("codex/"):
            raise ConfigurationError("Image and review models must use the same model family")
        max_cost_usd = _optional_money(
            "PAPERCLEAN_MAX_COST_USD",
            choose("max_cost_usd", "PAPERCLEAN_MAX_COST_USD", None),
        )
        zdr = _bool("PAPERCLEAN_ZDR", choose("zdr", "PAPERCLEAN_ZDR", False))
        if native_codex and max_cost_usd is not None:
            raise ConfigurationError("--max-cost-usd is unavailable with Codex subscription models")
        if native_codex and zdr:
            raise ConfigurationError("--zdr is unavailable with Codex subscription models")
        return cls(
            api_key=api_key,
            backend="agentbridge",
            base_url=base_url,
            image_model=image_model,
            review_model=review_model,
            max_attempts=_positive_int(
                "PAPERCLEAN_MAX_ATTEMPTS",
                choose("max_attempts", "PAPERCLEAN_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
            ),
            jobs=_positive_int("PAPERCLEAN_JOBS", choose("jobs", "PAPERCLEAN_JOBS", DEFAULT_JOBS)),
            max_cost_usd=max_cost_usd,
            zdr=zdr,
            agentbridge_timeout=_positive_int(
                "PAPERCLEAN_AGENTBRIDGE_TIMEOUT",
                choose(
                    "agentbridge_timeout",
                    "PAPERCLEAN_AGENTBRIDGE_TIMEOUT",
                    DEFAULT_AGENTBRIDGE_TIMEOUT,
                ),
            ),
        )


def gateway_model(model: str) -> str:
    """Keep the selected upstream model while adding the gateway namespace."""
    if (
        model.startswith(("codex/", "claudecode/"))
        or (model.startswith("openrouter/") and model.count("/") >= 2)
        or model == "openrouter/default"
    ):
        return model
    return f"openrouter/{model}"
