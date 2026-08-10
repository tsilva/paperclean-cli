from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from paperclean.agentbridge import AgentBridgeClient
from paperclean.config import Settings
from paperclean.errors import GlobalProviderError, ProviderError


def _png_bytes(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 20), color).save(output, format="PNG")
    return output.getvalue()


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "backend": "agentbridge",
        "review_enabled": True,
    }
    values.update(overrides)
    return Settings.from_sources(values, {})


def test_agentbridge_client_runs_preflight_generation_and_structured_review() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    generated = _png_bytes("white")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/capabilities":
            return httpx.Response(
                200,
                json={
                    "agentbridge_version": "0.1.9",
                    "codex": {
                        "available": True,
                        "authenticated": True,
                        "image_generation": True,
                        "json_schema": True,
                        "strict_profiles": True,
                    },
                },
            )
        if path == "/api/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "codex/gpt-5.6-sol", "object": "model"}]},
            )
        body = json.loads(request.content)
        calls.append((path, body))
        if path == "/api/v1/images":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"b64_json": base64.b64encode(generated).decode("ascii")}
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )
        if path == "/api/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "content_match": True,
                                        "scanner_quality": True,
                                        "discrepancies": [],
                                    }
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 1,
                        "total_tokens": 5,
                    },
                },
            )
        raise AssertionError(f"unexpected path: {path}")

    with AgentBridgeClient(
        _settings(), transport=httpx.MockTransport(handler)
    ) as client:
        client.preflight()
        projection = client.cost_projection(
            document_total=1,
            page_total=2,
            max_attempts=3,
        )
        candidate = client.generate(
            Image.new("RGB", (16, 20), "gray"),
            "Preserve all content and clean the page.",
            max_edge=4096,
        )
        verdict = client.review(
            Image.new("RGB", (16, 20), "gray"),
            candidate,
            view_name="full page",
        )

    assert candidate.size == (16, 20)
    assert verdict.accepted is True
    assert projection.billing_mode == "codex_subscription"
    assert projection.backend_version == "0.1.9"
    assert projection.one_pass.cost_usd is None
    assert projection.one_pass.generations == 2
    assert projection.one_pass.reviews == 10
    assert calls[0][0] == "/api/v1/images"
    assert calls[0][1]["store"] is False
    assert calls[0][1]["n"] == 1
    assert calls[1][0] == "/api/v1/chat/completions"
    assert calls[1][1]["store"] is False
    assert calls[1][1]["response_format"]["type"] == "json_schema"
    assert client.costs.prompt_tokens == 7
    assert client.costs.completion_tokens == 3
    assert client.costs.total_tokens == 10


def test_agentbridge_preflight_rejects_missing_strict_capability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/capabilities"
        return httpx.Response(
            200,
            json={
                "codex": {
                    "available": True,
                    "authenticated": True,
                    "image_generation": True,
                    "json_schema": True,
                    "strict_profiles": False,
                }
            },
        )

    with AgentBridgeClient(
        _settings(), transport=httpx.MockTransport(handler)
    ) as client, pytest.raises(GlobalProviderError, match="required authenticated"):
        client.preflight()


def test_agentbridge_generation_normalizes_invalid_image_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": "not-base64"}]})

    with AgentBridgeClient(
        _settings(review_enabled=False), transport=httpx.MockTransport(handler)
    ) as client, pytest.raises(ProviderError, match="invalid image"):
        client.generate(
            Image.new("RGB", (8, 8), "white"),
            "clean",
            max_edge=4096,
        )
