from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from paperclean.config import Settings
from paperclean.errors import GlobalOpenRouterError, PayloadTooLargeError
from paperclean.openrouter import CostTracker, OpenRouterClient


def _png_base64() -> str:
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def _settings(*, zdr: bool = False) -> Settings:
    return Settings(api_key="not-a-real-key", zdr=zdr)


def test_preflight_generation_and_review_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/images/models/openai/gpt-image-2/endpoints":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [
                            {
                                "provider_name": "OpenAI",
                                "provider_slug": "openai",
                                "supported_parameters": [
                                    "input_references",
                                    "quality",
                                    "background",
                                    "aspect_ratio",
                                ],
                            }
                        ]
                    }
                },
            )
        if request.url.path == "/api/v1/models/openai/gpt-5.6-sol/endpoints":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "architecture": {"input_modalities": ["text", "image"]},
                        "endpoints": [
                            {
                                "provider_name": "OpenAI",
                                "provider_slug": "openai",
                                "supported_parameters": ["response_format", "reasoning_effort"],
                            }
                        ],
                    }
                },
            )
        if request.url.path == "/api/v1/images":
            body = json.loads(request.content)
            assert body["provider"]["only"] == ["openai"]
            assert body["input_references"][0]["type"] == "image_url"
            assert body["input_references"][0]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
            return httpx.Response(
                200,
                json={"data": [{"b64_json": _png_base64()}], "usage": {"cost": 0.01}},
            )
        if request.url.path == "/api/v1/chat/completions":
            body = json.loads(request.content)
            assert body["response_format"]["json_schema"]["strict"] is True
            assert body["provider"] == {"only": ["openai"], "require_parameters": True}
            content = {"content_match": True, "scanner_quality": True, "discrepancies": []}
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": json.dumps(content)}}],
                    "usage": {"cost": 0.02, "total_tokens": 5},
                },
            )
        raise AssertionError(request.url)

    with OpenRouterClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        client.preflight()
        image = Image.new("RGB", (32, 32), "white")
        assert client.generate(image, "clean", max_edge=4096).size == (32, 32)
        assert client.review(image, image, view_name="full page").accepted
        assert float(client.costs.total) == pytest.approx(0.03)
    assert len(requests) == 4


def test_zdr_rejects_unlisted_pair() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("gpt-image-2/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [
                            {
                                "provider_slug": "openai",
                                "supported_parameters": ["input_references"],
                            }
                        ]
                    }
                },
            )
        if request.url.path.endswith("gpt-5.6-sol/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "architecture": {"input_modalities": ["image"]},
                        "endpoints": [
                            {"provider_slug": "openai", "supported_parameters": ["response_format"]}
                        ],
                    }
                },
            )
        if request.url.path.endswith("/endpoints/zdr"):
            return httpx.Response(200, json={"data": []})
        raise AssertionError(request.url)

    with (
        OpenRouterClient(_settings(zdr=True), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(GlobalOpenRouterError, match="does not support ZDR"),
    ):
        client.preflight()


def test_auth_failure_is_global() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(401, json={"error": {"message": "unauthorized"}})
    )
    with (
        OpenRouterClient(_settings(), transport=transport) as client,
        pytest.raises(GlobalOpenRouterError, match="authentication failed"),
    ):
        client.preflight()


def test_zdr_accepts_matching_model_tag_pair() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("gpt-image-2/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "endpoints": [
                            {"tag": "openai", "supported_parameters": ["input_references"]}
                        ]
                    }
                },
            )
        if request.url.path.endswith("gpt-5.6-sol/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "architecture": {"input_modalities": ["image"]},
                        "endpoints": [
                            {"tag": "openai", "supported_parameters": ["response_format"]}
                        ],
                    }
                },
            )
        if request.url.path.endswith("/endpoints/zdr"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"model_id": "openai/gpt-image-2", "tag": "openai"},
                        {"model_id": "openai/gpt-5.6-sol", "tag": "openai"},
                    ]
                },
            )
        raise AssertionError(request.url)

    with OpenRouterClient(_settings(zdr=True), transport=httpx.MockTransport(handler)) as client:
        client.preflight()


def test_invalid_cost_metadata_is_rejected() -> None:
    with pytest.raises(GlobalOpenRouterError, match="invalid cost"):
        CostTracker(None).record({"cost": "NaN"})


def test_image_too_large_400_is_typed_for_one_downscale_retry() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            400,
            json={"error": {"type": "image_too_large", "message": "too large"}},
        )
    )
    with (
        OpenRouterClient(_settings(), transport=transport) as client,
        pytest.raises(PayloadTooLargeError),
    ):
        client._request("POST", "/images", json_body={}, paid=True, retries=1)
