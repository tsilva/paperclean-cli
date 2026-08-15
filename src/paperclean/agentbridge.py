"""AgentBridge client for Codex-backed image generation and fidelity review."""

from __future__ import annotations

import base64
import json
import threading
from decimal import Decimal
from typing import Any, cast

import httpx
from PIL import Image

from paperclean.config import Settings
from paperclean.errors import (
    ContentPolicyError,
    GlobalProviderError,
    InputError,
    PayloadTooLargeError,
    ProviderError,
    ReviewerResponseError,
)
from paperclean.imaging import data_url, decode_bytes
from paperclean.models import PageGeometry, ReviewVerdict, UsageRecord
from paperclean.openrouter import (
    ORIENTATION_SCHEMA,
    PAGE_LOCATION_SCHEMA,
    REVIEW_SCHEMA,
    _parse_verdict,
)
from paperclean.preflight import CostProjection, build_subscription_projection
from paperclean.prompting import (
    CANDIDATE_QUALITY_PROMPT,
    ORIENTATION_PROMPT,
    PAGE_LOCATION_PROMPT,
    REVIEW_PROMPT,
    REVIEW_SYSTEM_PROMPT,
)

MAX_IMAGE_RESPONSE_BYTES = 32 * 1024 * 1024


class SubscriptionUsageTracker:
    """Track observable Codex orchestration tokens without inventing USD cost."""

    def __init__(self) -> None:
        self.total = Decimal("0")
        self.ambiguous_timeouts = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._lock = threading.Lock()

    def record(self, raw: dict[str, Any] | None) -> UsageRecord:
        usage = raw or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        prompt_value = prompt if isinstance(prompt, int) and prompt >= 0 else None
        completion_value = completion if isinstance(completion, int) and completion >= 0 else None
        total_value = total if isinstance(total, int) and total >= 0 else None
        with self._lock:
            self.prompt_tokens += prompt_value or 0
            self.completion_tokens += completion_value or 0
            self.total_tokens += total_value or (prompt_value or 0) + (completion_value or 0)
        return UsageRecord(
            cost_usd=None,
            prompt_tokens=prompt_value,
            completion_tokens=completion_value,
            total_tokens=total_value,
        )


class AgentBridgeClient:
    """Narrow synchronous client for PaperClean's AgentBridge contract."""

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.costs = SubscriptionUsageTracker()
        self.backend_version: str | None = None
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={"Content-Type": "application/json", "X-Title": "PaperClean"},
            timeout=httpx.Timeout(float(settings.agentbridge_timeout), connect=5.0),
            transport=transport,
        )

    def __enter__(self) -> AgentBridgeClient:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        global_failure: bool = False,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            timeout_error = ProviderError("AgentBridge request timed out", error_type="timeout")
            if global_failure:
                raise GlobalProviderError(str(timeout_error), error_type="timeout") from exc
            raise timeout_error from exc
        except httpx.HTTPError as exc:
            raise GlobalProviderError("cannot reach AgentBridge", error_type="network") from exc
        try:
            value = response.json()
        except ValueError as exc:
            raise GlobalProviderError("AgentBridge returned invalid JSON") from exc
        if response.status_code < 400:
            if not isinstance(value, dict):
                raise GlobalProviderError("AgentBridge returned an invalid response object")
            return value
        error_body = value.get("error") if isinstance(value, dict) else None
        error_type = (
            str(error_body.get("type") or error_body.get("code") or "http_error")
            if isinstance(error_body, dict)
            else "http_error"
        )
        safe_message = f"AgentBridge request failed with HTTP {response.status_code}"
        normalized = json.dumps(error_body).lower() if isinstance(error_body, dict) else ""
        if response.status_code == 413:
            raise PayloadTooLargeError(
                "AgentBridge rejected the image size",
                error_type=error_type,
                status_code=413,
            )
        if any(term in normalized for term in ("content policy", "moderation", "refusal")):
            raise ContentPolicyError(
                "Codex rejected this page under its content policy",
                error_type=error_type,
                status_code=response.status_code,
            )
        if global_failure or response.status_code in {400, 401, 403, 404}:
            raise GlobalProviderError(
                safe_message,
                error_type=error_type,
                status_code=response.status_code,
            )
        raise ProviderError(
            safe_message,
            error_type=error_type,
            status_code=response.status_code,
        )

    def preflight(self) -> None:
        capabilities = self._request("GET", "/capabilities", global_failure=True)
        self.backend_version = (
            str(capabilities.get("agentbridge_version"))
            if capabilities.get("agentbridge_version") is not None
            else None
        )
        codex = capabilities.get("codex")
        required = (
            "available",
            "authenticated",
            "image_generation",
            "json_schema",
            "strict_profiles",
        )
        if not isinstance(codex, dict) or not all(codex.get(key) is True for key in required):
            raise GlobalProviderError(
                "AgentBridge does not expose the required authenticated Codex capabilities"
            )
        models = self._request("GET", "/models", global_failure=True)
        rows = models.get("data")
        available = (
            {row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
            if isinstance(rows, list)
            else set()
        )
        selected = {self.settings.image_model, self.settings.review_model}
        if not selected <= available:
            raise GlobalProviderError("AgentBridge does not advertise the selected Codex model")

    def cost_projection(
        self, *, document_total: int, page_total: int, max_attempts: int
    ) -> CostProjection:
        return build_subscription_projection(
            document_total=document_total,
            page_total=page_total,
            max_attempts=max_attempts,
            image_model=self.settings.image_model,
            review_model=self.settings.review_model,
            backend_version=self.backend_version,
        )

    def generate(self, source: Image.Image, prompt: str, *, max_edge: int) -> Image.Image:
        response = self._request(
            "POST",
            "/images",
            json_body={
                "model": self.settings.image_model,
                "prompt": prompt,
                "input_references": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url(source, max_edge=max_edge)},
                    }
                ],
                "n": 1,
                "store": False,
            },
        )
        usage = response.get("usage")
        self.costs.record(usage if isinstance(usage, dict) else None)
        rows = response.get("data")
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ProviderError("AgentBridge image generation returned no image")
        encoded = rows[0].get("b64_json")
        if not isinstance(encoded, str):
            raise ProviderError("AgentBridge returned an unsupported image payload")
        if len(encoded) > ((MAX_IMAGE_RESPONSE_BYTES + 2) // 3) * 4 + 4:
            raise ProviderError("AgentBridge returned an oversized image payload")
        try:
            decoded = base64.b64decode(encoded, validate=True)
            if len(decoded) > MAX_IMAGE_RESPONSE_BYTES:
                raise ValueError("decoded image exceeds the response limit")
            return decode_bytes(decoded)
        except (InputError, ValueError) as exc:
            raise ProviderError("AgentBridge returned invalid image base64") from exc

    def locate_page(self, source: Image.Image) -> PageGeometry | None:
        response = self._request(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.settings.review_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Analyze document geometry conservatively.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PAGE_LOCATION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url(source, max_edge=2048)},
                            },
                        ],
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": PAGE_LOCATION_SCHEMA,
                },
                "reasoning_effort": "medium",
                "max_tokens": 1024,
                "store": False,
            },
        )
        usage_raw = response.get("usage")
        self.costs.record(usage_raw if isinstance(usage_raw, dict) else None)
        try:
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content) if isinstance(content, str) else content
            if not isinstance(value, dict) or value.get("found") is not True:
                return None
            confidence = float(value["confidence"])
            rows = value["corners"]
            page_polygon_rows = value["page_polygon"]
            content_rows = value["content_corners"]
            edge_content_rows = value["edge_content"]
            occlusion_rows = value["occlusions"]
            if (
                not isinstance(rows, list)
                or len(rows) != 4
                or not isinstance(page_polygon_rows, list)
                or not 4 <= len(page_polygon_rows) <= 40
                or not isinstance(content_rows, list)
                or len(content_rows) != 4
                or not isinstance(edge_content_rows, list)
                or not isinstance(occlusion_rows, list)
            ):
                raise ValueError("invalid corner count")
            corner_type = tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ]
            corners = cast(corner_type, tuple((float(row[0]), float(row[1])) for row in rows))
            page_polygon = tuple((float(row[0]), float(row[1])) for row in page_polygon_rows)
            content_corners = cast(
                corner_type,
                tuple((float(row[0]), float(row[1])) for row in content_rows),
            )
            edge_content = tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in edge_content_rows
            )
            occlusions = tuple(
                tuple((float(point[0]), float(point[1])) for point in polygon)
                for polygon in occlusion_rows
            )
            if any(
                not 0 <= coordinate <= 1
                for point in (
                    *corners,
                    *page_polygon,
                    *content_corners,
                    *(point for polygon in edge_content for point in polygon),
                    *(point for polygon in occlusions for point in polygon),
                )
                for coordinate in point
            ):
                raise ValueError("corner out of bounds")
            return PageGeometry(
                corners=corners,
                content_corners=content_corners,
                occlusions=occlusions,
                confidence=confidence,
                page_polygon=page_polygon,
                edge_content=edge_content,
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReviewerResponseError(
                "AgentBridge returned invalid structured page geometry"
            ) from exc

    def reading_rotation(self, source: Image.Image) -> int:
        response = self._request(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.settings.review_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Classify document reading orientation.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": ORIENTATION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url(source, max_edge=2048)},
                            },
                        ],
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": ORIENTATION_SCHEMA,
                },
                "reasoning_effort": "medium",
                "max_tokens": 512,
                "store": False,
            },
        )
        usage_raw = response.get("usage")
        self.costs.record(usage_raw if isinstance(usage_raw, dict) else None)
        try:
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content) if isinstance(content, str) else content
            rotation = int(value["rotation_degrees"])
            confidence = float(value["confidence"])
            if rotation not in {0, 90, 180, 270} or not 0 <= confidence <= 1:
                raise ValueError("invalid orientation")
            return rotation if confidence >= 0.90 else 0
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReviewerResponseError(
                "AgentBridge returned invalid structured reading orientation"
            ) from exc

    def review(
        self, source: Image.Image, candidate: Image.Image, *, view_name: str
    ) -> ReviewVerdict:
        prompt = REVIEW_PROMPT.replace("{view_name}", view_name)
        response = self._request(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.settings.review_model,
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
                "reasoning_effort": "medium",
                "max_tokens": 2048,
                "store": False,
            },
        )
        usage_raw = response.get("usage")
        usage = self.costs.record(usage_raw if isinstance(usage_raw, dict) else None)
        try:
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content) if isinstance(content, str) else content
            return _parse_verdict(value, usage)
        except ReviewerResponseError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReviewerResponseError(
                "AgentBridge returned invalid structured review output"
            ) from exc

    def review_quality(self, candidate: Image.Image, *, view_name: str) -> ReviewVerdict:
        prompt = CANDIDATE_QUALITY_PROMPT.replace("{view_name}", view_name)
        response = self._request(
            "POST",
            "/chat/completions",
            json_body={
                "model": self.settings.review_model,
                "messages": [
                    {
                        "role": "system",
                        "content": REVIEW_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "text", "text": "CANDIDATE:"},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url(candidate, max_edge=2048)},
                            },
                        ],
                    },
                ],
                "response_format": {"type": "json_schema", "json_schema": REVIEW_SCHEMA},
                "reasoning_effort": "medium",
                "max_tokens": 2048,
                "store": False,
            },
        )
        usage_raw = response.get("usage")
        usage = self.costs.record(usage_raw if isinstance(usage_raw, dict) else None)
        try:
            content = response["choices"][0]["message"]["content"]
            value = json.loads(content) if isinstance(content, str) else content
            verdict = _parse_verdict(value, usage)
        except ReviewerResponseError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReviewerResponseError(
                "AgentBridge returned invalid structured quality review output"
            ) from exc
        if not verdict.content_match or any(
            item.category != "scanner_quality" for item in verdict.discrepancies
        ):
            raise ReviewerResponseError("AgentBridge quality review violated its contract")
        return verdict
