"""Typed failures used to preserve conservative processing semantics."""

from __future__ import annotations

_PUBLIC_PROVIDER_ERROR_TYPES = frozenset(
    {
        "content_policy",
        "content_policy_violation",
        "generation_error",
        "http_error",
        "image_too_large",
        "missing_image",
        "moderation",
        "network",
        "no_output",
        "no_output_error",
        "refusal",
        "safety",
        "timeout",
        "timeout_error",
    }
)


class PaperCleanError(Exception):
    """Base class for expected PaperClean failures."""


class ConfigurationError(PaperCleanError):
    """Configuration or environment is unusable."""


class InputError(PaperCleanError):
    """An input path or document is unsupported or invalid."""


class OutputCollisionError(PaperCleanError):
    """A target output or its report already exists."""


class UnsafePdfError(InputError):
    """The PDF contains active or redaction content that v1 refuses to transform."""


class ProviderError(PaperCleanError):
    """Base class for normalized model-provider failures."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        # Provider codes can reflect upstream credentials unknown to this client.
        # Only fixed, recognized labels may reach reports or public diagnostics.
        self.error_type = (
            error_type
            if error_type is None or error_type in _PUBLIC_PROVIDER_ERROR_TYPES
            else "provider_error"
        )
        self.status_code = status_code


class GlobalProviderError(ProviderError):
    """A systemic provider failure that must stop scheduling new documents."""


class OpenRouterError(ProviderError):
    """Backwards-compatible OpenRouter-specific provider failure."""


class GlobalOpenRouterError(GlobalProviderError, OpenRouterError):
    """A systemic failure that must stop scheduling new documents."""


class ContentPolicyError(OpenRouterError):
    """A page-specific moderation or content-policy failure."""


class PayloadTooLargeError(OpenRouterError):
    """The provider refused the image payload size."""


class ReviewerResponseError(OpenRouterError):
    """The reviewer refused or returned invalid structured output."""


class CostLimitReached(PaperCleanError):
    """The observed soft cost ceiling was reached."""


class CostUnavailableError(GlobalOpenRouterError):
    """A cost ceiling was requested but the provider omitted cost metadata."""
