"""Narrow, typed OpenRouter client for generation and fidelity review."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from typing import Any, Literal, cast

import httpx
from PIL import Image

from paperclean.config import Settings
from paperclean.errors import (
    ContentPolicyError,
    CostLimitReached,
    CostUnavailableError,
    GlobalOpenRouterError,
    OpenRouterError,
    PayloadTooLargeError,
    ReviewerResponseError,
)
from paperclean.imaging import data_url, decode_bytes, decode_data_url
from paperclean.models import Discrepancy, ReviewVerdict, UsageRecord
from paperclean.preflight import (
    REVIEW_MAX_COMPLETION_TOKENS,
    CostProjection,
    UnitPrices,
    build_cost_projection,
    parse_unit_prices,
)
from paperclean.prompting import REVIEW_PROMPT, REVIEW_SYSTEM_PROMPT

ALLOWED_CATEGORIES = (
    "changed_text",
    "missing_text",
    "invented_text",
    "changed_handwriting",
    "changed_signature",
    "changed_stamp",
    "changed_redaction",
    "changed_table",
    "changed_diagram",
    "changed_layout",
    "cropped_content",
    "scanner_quality",
    "unresolved_content",
    "other_content",
)
ALLOWED_SEVERITIES = ("low", "medium", "high", "critical")
PAYMENT_ERROR_TERMS = (
    "afford",
    "balance",
    "credit",
    "fund",
    "max_token",
    "payment",
    "spend limit",
    "usage limit",
)
GENERIC_PAYMENT_ERROR = "OpenRouter credits or payment are required"

REVIEW_SCHEMA: dict[str, Any] = {
    "name": "paperclean_fidelity_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["content_match", "scanner_quality", "discrepancies"],
        "properties": {
            "content_match": {"type": "boolean"},
            "scanner_quality": {"type": "boolean"},
            "discrepancies": {
                "type": "array",
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["category", "severity", "region"],
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": list(ALLOWED_CATEGORIES),
                        },
                        "severity": {
                            "type": "string",
                            "enum": list(ALLOWED_SEVERITIES),
                        },
                        "region": {
                            "type": "array",
                            "items": {"type": "number", "minimum": 0, "maximum": 1},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                },
            },
        },
    },
}


@dataclass(frozen=True, slots=True)
class Endpoint:
    model: str
    provider_name: str
    provider_slug: str
    supported_parameters: frozenset[str]
    prices: UnitPrices


class CostTracker:
    def __init__(self, limit: Decimal | None) -> None:
        self.limit = limit
        self.total = Decimal("0")
        self.ambiguous_timeouts = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._lock = threading.Lock()

    def before_request(self) -> None:
        with self._lock:
            if self.limit is not None and self.total >= self.limit:
                raise CostLimitReached(f"soft cost limit of ${self.limit} reached")

    def record(self, usage: dict[str, Any] | None) -> UsageRecord:
        usage = usage or {}
        raw_cost = usage.get("cost")
        cost: Decimal | None = None
        if raw_cost is not None:
            try:
                cost = Decimal(str(raw_cost))
            except (InvalidOperation, ValueError) as exc:
                raise GlobalOpenRouterError("OpenRouter returned invalid cost metadata") from exc
            if not cost.is_finite() or cost < 0:
                raise GlobalOpenRouterError("OpenRouter returned invalid cost metadata")
        with self._lock:
            if self.limit is not None and raw_cost is None:
                raise CostUnavailableError(
                    "OpenRouter omitted cost metadata required by --max-cost-usd"
                )
            if cost is not None:
                self.total += cost
            prompt_tokens = _as_int(usage.get("prompt_tokens"))
            completion_tokens = _as_int(usage.get("completion_tokens"))
            total_tokens = _as_int(usage.get("total_tokens"))
            self.prompt_tokens += prompt_tokens or 0
            self.completion_tokens += completion_tokens or 0
            self.total_tokens += total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)
        return UsageRecord(
            cost_usd=float(cost) if cost is not None else None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def ambiguous_timeout(self) -> None:
        with self._lock:
            self.ambiguous_timeouts += 1


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _model_parts(model: str) -> tuple[str, str]:
    author, slug = model.split("/", 1)
    return author, slug


def _retry_after(response: httpx.Response) -> float:
    value = response.headers.get("retry-after")
    if value is None:
        return 1.0
    try:
        return min(30.0, max(0.0, float(value)))
    except ValueError:
        try:
            return min(30.0, max(0.0, parsedate_to_datetime(value).timestamp() - time.time()))
        except (TypeError, ValueError):
            return 1.0


class OpenRouterClient:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.backend_version: str | None = None
        self.costs = CostTracker(settings.max_cost_usd)
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/tsilva/paperclean",
                "X-Title": "PaperClean",
            },
            timeout=httpx.Timeout(180.0, connect=20.0),
            transport=transport,
        )
        self.image_endpoint: Endpoint | None = None
        self.review_endpoint: Endpoint | None = None

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        paid: bool = False,
        retries: int = 3,
    ) -> dict[str, Any]:
        if paid:
            self.costs.before_request()
        for attempt in range(retries):
            try:
                response = self._client.request(method, path, json=json_body)
            except httpx.TimeoutException as exc:
                if paid:
                    self.costs.ambiguous_timeout()
                if attempt + 1 < retries:
                    time.sleep(min(4.0, 2**attempt))
                    continue
                raise GlobalOpenRouterError(
                    "OpenRouter request timed out", error_type="timeout"
                ) from exc
            except httpx.HTTPError as exc:
                raise GlobalOpenRouterError(
                    "cannot reach OpenRouter", error_type="network"
                ) from exc
            if response.status_code < 400:
                try:
                    value = response.json()
                except ValueError as exc:
                    raise GlobalOpenRouterError("OpenRouter returned invalid JSON") from exc
                if not isinstance(value, dict):
                    raise GlobalOpenRouterError("OpenRouter returned an invalid response object")
                embedded_error = value.get("error")
                if isinstance(embedded_error, dict):
                    if paid:
                        usage = value.get("usage")
                        self.costs.record(usage if isinstance(usage, dict) else None)
                    error_type = str(
                        embedded_error.get("type")
                        or embedded_error.get("code")
                        or "generation_error"
                    )
                    normalized = error_type.lower()
                    if any(
                        item in normalized
                        for item in ("moderation", "content_policy", "refusal", "safety")
                    ):
                        raise ContentPolicyError(
                            "the provider rejected this page under its content policy",
                            error_type=error_type,
                        )
                    raise OpenRouterError(
                        "the provider failed while producing a result",
                        error_type=error_type,
                    )
                return value
            error = _error_data(response)
            error_type = str(error.get("type") or error.get("code") or "http_error")
            message = str(error.get("message") or f"OpenRouter HTTP {response.status_code}")
            transient = response.status_code in {408, 409, 429} or response.status_code >= 500
            if transient and attempt + 1 < retries:
                time.sleep(
                    _retry_after(response) if response.status_code == 429 else min(4.0, 2**attempt)
                )
                continue
            normalized = (error_type + " " + message).lower()
            if response.status_code == 413 or any(
                item in normalized for item in ("image_too_large", "too large")
            ):
                raise PayloadTooLargeError(message, error_type=error_type, status_code=413)
            if response.status_code in {400, 422} and any(
                item in normalized
                for item in (
                    "moderation",
                    "content policy",
                    "content_policy",
                    "refusal",
                    "safety",
                )
            ):
                raise ContentPolicyError(
                    message, error_type=error_type, status_code=response.status_code
                )
            if response.status_code in {401, 402, 403, 404}:
                safe_messages = {
                    401: "OpenRouter authentication failed",
                    402: _safe_payment_error(message),
                    403: "OpenRouter permission denied",
                    404: "OpenRouter model or endpoint was not found",
                }
                raise GlobalOpenRouterError(
                    safe_messages[response.status_code],
                    error_type=error_type,
                    status_code=response.status_code,
                )
            raise OpenRouterError(message, error_type=error_type, status_code=response.status_code)
        raise AssertionError("unreachable")

    def _select_endpoint(
        self,
        model: str,
        *,
        images_api: bool,
        predicate: Callable[[frozenset[str], dict[str, Any]], bool],
    ) -> Endpoint:
        author, slug = _model_parts(model)
        prefix = "/images" if images_api else ""
        response = self._request("GET", f"{prefix}/models/{author}/{slug}/endpoints")
        data = response.get("data", response)
        endpoint_rows = data.get("endpoints", []) if isinstance(data, dict) else []
        architecture = data.get("architecture", {}) if isinstance(data, dict) else {}
        for row in endpoint_rows:
            if not isinstance(row, dict):
                continue
            parameters = frozenset(str(item) for item in row.get("supported_parameters", []))
            if predicate(parameters, architecture if isinstance(architecture, dict) else {}):
                name = str(row.get("provider_name") or row.get("provider") or row.get("name") or "")
                slug_value = str(row.get("provider_slug") or row.get("tag") or name)
                if slug_value:
                    return Endpoint(
                        model,
                        name or slug_value,
                        slug_value,
                        parameters,
                        parse_unit_prices(row.get("pricing")),
                    )
        kind = "image reference generation" if images_api else "image review with structured output"
        raise GlobalOpenRouterError(f"no endpoint for {model} supports {kind}")

    def preflight(self) -> None:
        self.image_endpoint = self._select_endpoint(
            self.settings.image_model,
            images_api=True,
            predicate=lambda params, _arch: "input_references" in params,
        )
        self.review_endpoint = self._select_endpoint(
            self.settings.review_model,
            images_api=False,
            predicate=lambda params, arch: (
                "image" in set(arch.get("input_modalities", []))
                and bool({"response_format", "structured_outputs"} & params)
                and bool({"max_tokens", "max_completion_tokens"} & params)
            ),
        )
        if self.settings.zdr:
            response = self._request("GET", "/endpoints/zdr")
            rows = response.get("data", [])
            allowed = {
                (
                    str(row.get("model_id") or row.get("model")),
                    str(row.get("provider_slug") or row.get("tag") or row.get("provider")),
                )
                for row in rows
                if isinstance(row, dict)
            }
            selected = {(self.image_endpoint.model, self.image_endpoint.provider_slug)}
            selected.add((self.review_endpoint.model, self.review_endpoint.provider_slug))
            missing = selected - allowed
            if missing:
                raise GlobalOpenRouterError("the selected model/provider pair does not support ZDR")

    def cost_projection(
        self, *, document_total: int, page_total: int, max_attempts: int
    ) -> CostProjection:
        image_endpoint = self._required_endpoint(self.image_endpoint, "preflight image endpoint")
        review_endpoint = self._required_endpoint(self.review_endpoint, "preflight review endpoint")
        account_remaining = self._account_remaining()
        key_remaining, key_unlimited = self._key_remaining()
        return build_cost_projection(
            document_total=document_total,
            page_total=page_total,
            max_attempts=max_attempts,
            image_model=image_endpoint.model,
            image_provider=image_endpoint.provider_name,
            image_prices=image_endpoint.prices,
            review_model=review_endpoint.model,
            review_provider=review_endpoint.provider_name,
            review_prices=review_endpoint.prices,
            account_remaining_usd=account_remaining,
            key_remaining_usd=key_remaining,
            key_unlimited=key_unlimited,
            soft_limit_usd=self.settings.max_cost_usd,
        )

    def _optional_metadata(self, path: str) -> dict[str, Any] | None:
        try:
            return self._request("GET", path)
        except GlobalOpenRouterError as exc:
            if exc.status_code in {403, 404}:
                return None
            raise

    @staticmethod
    def _metadata_decimal(value: Any, description: str) -> Decimal | None:
        if value is None:
            return None
        try:
            result = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise GlobalOpenRouterError(
                f"OpenRouter returned invalid {description} metadata"
            ) from exc
        if not result.is_finite() or result < 0:
            raise GlobalOpenRouterError(f"OpenRouter returned invalid {description} metadata")
        return result

    def _account_remaining(self) -> Decimal | None:
        response = self._optional_metadata("/credits")
        if response is None:
            return None
        data = response.get("data")
        if not isinstance(data, dict):
            return None
        total = self._metadata_decimal(data.get("total_credits"), "credit")
        used = self._metadata_decimal(data.get("total_usage"), "credit")
        if total is None or used is None:
            return None
        return max(Decimal("0"), total - used)

    def _key_remaining(self) -> tuple[Decimal | None, bool]:
        response = self._optional_metadata("/key")
        if response is None:
            return None, False
        data = response.get("data")
        if not isinstance(data, dict):
            return None, False
        limit = self._metadata_decimal(data.get("limit"), "key limit")
        remaining = self._metadata_decimal(data.get("limit_remaining"), "key limit")
        return remaining, limit is None and "limit" in data and data["limit"] is None

    def generate(self, source: Image.Image, prompt: str, *, max_edge: int) -> Image.Image:
        endpoint = self._required_endpoint(self.image_endpoint, "preflight image endpoint")
        body: dict[str, Any] = {
            "model": endpoint.model,
            "prompt": prompt,
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url(source, max_edge=max_edge)},
                }
            ],
            "n": 1,
            "provider": {"only": [endpoint.provider_slug]},
        }
        if "quality" in endpoint.supported_parameters:
            body["quality"] = "high"
        if "background" in endpoint.supported_parameters:
            body["background"] = "opaque"
        if "aspect_ratio" in endpoint.supported_parameters:
            body["aspect_ratio"] = "auto"
        response = self._request("POST", "/images", json_body=body, paid=True)
        self.costs.record(
            response.get("usage") if isinstance(response.get("usage"), dict) else None
        )
        rows = response.get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise OpenRouterError("image generation returned no image", error_type="missing_image")
        item = rows[0]
        raw = item.get("b64_json")
        if isinstance(raw, str):
            import base64

            try:
                return decode_bytes(base64.b64decode(raw, validate=True))
            except ValueError as exc:
                raise OpenRouterError("image generation returned invalid base64") from exc
        url = item.get("url") or item.get("image_url")
        if isinstance(url, str) and url.startswith("data:"):
            return decode_bytes(decode_data_url(url))
        raise OpenRouterError("image generation returned an unsupported image payload")

    def review(
        self, source: Image.Image, candidate: Image.Image, *, view_name: str
    ) -> ReviewVerdict:
        endpoint = self._required_endpoint(self.review_endpoint, "preflight review endpoint")
        prompt = REVIEW_PROMPT.replace("{view_name}", view_name)
        body: dict[str, Any] = {
            "model": endpoint.model,
            "messages": [
                {
                    "role": "system",
                    "content": REVIEW_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "text", "text": "ORIGINAL:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(source, max_edge=2048)},
                        },
                        {"type": "text", "text": "CANDIDATE:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url(candidate, max_edge=2048)},
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_schema", "json_schema": REVIEW_SCHEMA},
            "provider": {"only": [endpoint.provider_slug], "require_parameters": True},
        }
        if "reasoning_effort" in endpoint.supported_parameters:
            # Leave enough of the completion budget for the structured verdict.
            # High effort can consume the entire budget as hidden reasoning.
            body["reasoning_effort"] = "medium"
        if "max_completion_tokens" in endpoint.supported_parameters:
            body["max_completion_tokens"] = REVIEW_MAX_COMPLETION_TOKENS
        else:
            body["max_tokens"] = REVIEW_MAX_COMPLETION_TOKENS
        response = self._request("POST", "/chat/completions", json_body=body, paid=True)
        usage = self.costs.record(
            response.get("usage") if isinstance(response.get("usage"), dict) else None
        )
        try:
            message = response["choices"][0]["message"]
            if message.get("refusal"):
                raise ReviewerResponseError("the reviewer refused the comparison")
            content = message["content"]
            value = json.loads(content) if isinstance(content, str) else content
            return _parse_verdict(value, usage)
        except ReviewerResponseError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReviewerResponseError("the reviewer returned invalid structured output") from exc

    @staticmethod
    def _required_endpoint(endpoint: Endpoint | None, message: str) -> Endpoint:
        if endpoint is None:
            raise GlobalOpenRouterError(message)
        return endpoint


def _error_data(response: httpx.Response) -> dict[str, Any]:
    try:
        value = response.json()
    except ValueError:
        return {}
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        error = value["error"]
        return cast(dict[str, Any], error)
    return value if isinstance(value, dict) else {}


def _safe_payment_error(message: str) -> str:
    """Retain actionable billing detail without echoing arbitrary provider content."""
    detail = " ".join(message.split())
    normalized = detail.lower()
    if (
        not detail
        or len(detail) > 500
        or not all(character.isprintable() for character in detail)
        or not any(term in normalized for term in PAYMENT_ERROR_TERMS)
    ):
        return GENERIC_PAYMENT_ERROR
    return f"OpenRouter payment required: {detail}"


def _parse_verdict(value: Any, usage: UsageRecord) -> ReviewVerdict:
    if not isinstance(value, dict):
        raise ReviewerResponseError("review verdict is not an object")
    if not isinstance(value.get("content_match"), bool) or not isinstance(
        value.get("scanner_quality"), bool
    ):
        raise ReviewerResponseError("review verdict booleans are missing")
    rows = value.get("discrepancies")
    if not isinstance(rows, list):
        raise ReviewerResponseError("review discrepancies are missing")
    discrepancies: list[Discrepancy] = []
    categories = set(ALLOWED_CATEGORIES)
    severities = set(ALLOWED_SEVERITIES)
    for row in rows:
        if not isinstance(row, dict):
            raise ReviewerResponseError("review discrepancy is invalid")
        region = row.get("region")
        if not isinstance(region, list) or len(region) != 4:
            raise ReviewerResponseError("review discrepancy region is invalid")
        numbers = tuple(float(item) for item in region)
        if any(not math.isfinite(item) or item < 0 or item > 1 for item in numbers):
            raise ReviewerResponseError("review discrepancy region is out of range")
        category = row.get("category")
        severity = row.get("severity")
        if category not in categories or severity not in severities:
            raise ReviewerResponseError("review discrepancy category is invalid")
        discrepancies.append(
            Discrepancy(
                category=str(category),
                severity=cast(Literal["low", "medium", "high", "critical"], severity),
                region=cast(tuple[float, float, float, float], numbers),
            )
        )
    return ReviewVerdict(
        content_match=value["content_match"],
        scanner_quality=value["scanner_quality"],
        discrepancies=discrepancies,
        usage=usage,
    )
