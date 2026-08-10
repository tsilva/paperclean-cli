from __future__ import annotations

import base64
import io
import json
from decimal import Decimal

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


def _settings(*, zdr: bool = False, review_enabled: bool = True) -> Settings:
    return Settings(api_key="not-a-real-key", zdr=zdr, review_enabled=review_enabled)


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
                                "supported_parameters": [
                                    "response_format",
                                    "reasoning_effort",
                                    "max_tokens",
                                ],
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
            assert body["reasoning_effort"] == "medium"
            assert body["max_tokens"] == 4096
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
                            {
                                "provider_slug": "openai",
                                "supported_parameters": ["response_format", "max_tokens"],
                            }
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
                            {
                                "tag": "openai",
                                "supported_parameters": ["response_format", "max_tokens"],
                            }
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


def test_cost_projection_uses_selected_endpoint_prices_and_credit_balances() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("gpt-image-2/endpoints"):
            return httpx.Response(
                200,
                json={
                    "endpoints": [
                        {
                            "provider_name": "OpenAI",
                            "provider_slug": "openai",
                            "supported_parameters": ["input_references"],
                            "pricing": [
                                {
                                    "billable": "input_image",
                                    "unit": "token",
                                    "cost_usd": 0.000008,
                                },
                                {
                                    "billable": "input_text",
                                    "unit": "token",
                                    "cost_usd": 0.000005,
                                },
                                {
                                    "billable": "output_image",
                                    "unit": "token",
                                    "cost_usd": 0.00003,
                                },
                            ],
                        }
                    ]
                },
            )
        if request.url.path.endswith("gpt-5.6-sol/endpoints"):
            return httpx.Response(
                200,
                json={
                    "data": {
                        "architecture": {"input_modalities": ["text", "image"]},
                        "endpoints": [
                            {
                                "provider_name": "OpenAI",
                                "provider_slug": "openai",
                                "supported_parameters": ["response_format", "max_tokens"],
                                "pricing": {"prompt": "0.000005", "completion": "0.00003"},
                            }
                        ],
                    }
                },
            )
        if request.url.path.endswith("/credits"):
            return httpx.Response(
                200,
                json={"data": {"total_credits": 509.11, "total_usage": 508.391410008}},
            )
        if request.url.path.endswith("/key"):
            return httpx.Response(
                200,
                json={"data": {"limit": 10, "limit_remaining": 10}},
            )
        raise AssertionError(request.url)

    with OpenRouterClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        client.preflight()
        projection = client.cost_projection(document_total=1, page_total=1, max_attempts=3)

    assert projection.one_pass.generations == 1
    assert projection.one_pass.reviews == 5
    assert projection.one_pass.paid_calls == 6
    assert projection.one_pass.cost_usd == Decimal("1.189452")
    assert projection.configured_max.cost_usd == Decimal("3.568356")
    assert projection.recovery_ceiling.paid_calls == 36
    assert projection.recovery_ceiling.cost_usd == Decimal("7.136712")
    assert projection.account_remaining_usd == Decimal("0.718589992")
    assert projection.key_remaining_usd == Decimal("10")


def test_disabled_review_skips_endpoint_and_projects_generation_only() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("gpt-image-2/endpoints"):
            return httpx.Response(
                200,
                json={
                    "endpoints": [
                        {
                            "provider_name": "OpenAI",
                            "provider_slug": "openai",
                            "supported_parameters": ["input_references"],
                            "pricing": [
                                {
                                    "billable": "input_image",
                                    "unit": "token",
                                    "cost_usd": 0.000008,
                                },
                                {
                                    "billable": "input_text",
                                    "unit": "token",
                                    "cost_usd": 0.000005,
                                },
                                {
                                    "billable": "output_image",
                                    "unit": "token",
                                    "cost_usd": 0.00003,
                                },
                            ],
                        }
                    ]
                },
            )
        if request.url.path.endswith("/endpoints/zdr"):
            return httpx.Response(
                200,
                json={"data": [{"model_id": "openai/gpt-image-2", "provider_slug": "openai"}]},
            )
        if request.url.path.endswith("/credits"):
            return httpx.Response(200, json={"data": {"total_credits": 10, "total_usage": 0}})
        if request.url.path.endswith("/key"):
            return httpx.Response(200, json={"data": {"limit": None, "limit_remaining": None}})
        raise AssertionError(request.url)

    settings = _settings(zdr=True, review_enabled=False)
    with OpenRouterClient(settings, transport=httpx.MockTransport(handler)) as client:
        client.preflight()
        projection = client.cost_projection(document_total=1, page_total=1, max_attempts=3)

    assert not any("gpt-5.6-sol" in path for path in paths)
    assert projection.review_enabled is False
    assert projection.review_model is None
    assert projection.review_provider is None
    assert projection.one_pass.reviews == 0
    assert projection.one_pass.paid_calls == 1
    assert projection.one_pass.cost_usd == Decimal("0.344652")
    assert projection.configured_max.cost_usd == Decimal("1.033956")
    assert projection.recovery_ceiling.paid_calls == 6
    assert projection.recovery_ceiling.cost_usd == Decimal("2.067912")


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


def test_payment_error_preserves_short_billing_detail() -> None:
    detail = (
        "This request requires more credits, or fewer max_tokens. "
        "You requested up to 16384 tokens, but can only afford 3594."
    )
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(402, json={"error": {"message": detail}})
    )
    with (
        OpenRouterClient(_settings(), transport=transport) as client,
        pytest.raises(GlobalOpenRouterError) as raised,
    ):
        client._request("POST", "/images", json_body={}, paid=True, retries=1)

    assert str(raised.value) == f"OpenRouter payment required: {detail}"
    assert raised.value.status_code == 402


def test_payment_error_does_not_echo_unrelated_provider_content() -> None:
    unrelated = "private document content returned by an upstream provider"
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(402, json={"error": {"message": unrelated}})
    )
    with (
        OpenRouterClient(_settings(), transport=transport) as client,
        pytest.raises(GlobalOpenRouterError) as raised,
    ):
        client._request("POST", "/images", json_body={}, paid=True, retries=1)

    assert str(raised.value) == "OpenRouter credits or payment are required"
    assert unrelated not in str(raised.value)
