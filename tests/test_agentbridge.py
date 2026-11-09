from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from paperclean.agentbridge import AgentBridgeClient
from paperclean.config import Settings
from paperclean.discovery import output_paths
from paperclean.errors import (
    ContentPolicyError,
    GlobalProviderError,
    PayloadTooLargeError,
    ProviderError,
)
from paperclean.models import AttemptRecord, PageRecord
from paperclean.pipeline import _base_report, _error_name
from paperclean.provenance import write_report


def _png_bytes(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 20), color).save(output, format="PNG")
    return output.getvalue()


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "image_model": "codex/gpt-6.1-sol",
        "review_model": "codex/gpt-6.1-sol",
    }
    values.update(overrides)
    return Settings.from_sources(values, {})


@pytest.mark.parametrize("model", ["codex/gpt-6.1-sol", "openai/gpt-image-2"])
@pytest.mark.parametrize("gateway_key", ["", "dummy-gateway-placeholder"])
def test_gateway_authentication_header_is_optional(model: str, gateway_key: str) -> None:
    settings = Settings.from_sources(
        {"image_model": model, "review_model": model},
        {"AGENTBRIDGE_API_KEY": gateway_key},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if gateway_key:
            assert request.headers["Authorization"] == f"Bearer {gateway_key}"
        else:
            assert "Authorization" not in request.headers
        return httpx.Response(200, json={})

    with AgentBridgeClient(settings, transport=httpx.MockTransport(handler)) as client:
        client._client.get("/capabilities")


@pytest.mark.parametrize("model", ["codex/gpt-6.1-sol", "openai/gpt-image-2"])
@pytest.mark.parametrize("status", [422, 400, 200, 402])
@pytest.mark.parametrize("code_field", ["type", "code"])
def test_provider_errors_cannot_reflect_credentials_into_diagnostics_or_reports(
    model: str, status: int, code_field: str, tmp_path: Path
) -> None:
    gateway_key = "dummy-gateway-placeholder"
    upstream_key = "dummy-upstream-credential-unknown-to-client"
    reflected = f"credits: {gateway_key} {upstream_key}"
    settings = Settings.from_sources(
        {"image_model": model, "review_model": model}, {"AGENTBRIDGE_API_KEY": gateway_key}
    )
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            status, json={"error": {code_field: reflected, "message": reflected}}
        )
    )
    with (
        AgentBridgeClient(settings, transport=transport) as client,
        pytest.raises(ProviderError) as raised,
    ):
        if client._metered:
            client._metered._request("POST", "/images", json_body={}, retries=1)
        else:
            client._request("POST", "/images", json_body={})

    error = raised.value
    assert error.status_code == status
    assert error.error_type == "provider_error"
    assert _error_name(error) == "provider_error"
    source = tmp_path / "input.png"
    source.write_bytes(_png_bytes())
    report = _base_report(output_paths(source), settings)
    report.pages.append(
        PageRecord(
            page=1,
            status="original_fallback",
            source_render_sha256="source",
            final_render_sha256="source",
            attempts=[AttemptRecord(number=1, error_type=_error_name(error))],
        )
    )
    report_path = tmp_path / "report.json"
    write_report(report_path, report.as_dict())
    public_output = str(error) + json.dumps(error.__dict__) + report_path.read_text()
    assert gateway_key not in public_output
    assert upstream_key not in public_output
    assert json.loads(report_path.read_text())["pages"][0]["attempts"][0]["error_type"] == (
        "provider_error"
    )


@pytest.mark.parametrize("model", ["codex/gpt-6.1-sol", "openai/gpt-image-2"])
@pytest.mark.parametrize(
    ("status", "error_type", "error_class"),
    [
        (422, "timeout_error", ProviderError),
        (422, "no_output", ProviderError),
        (200, "moderation", ContentPolicyError),
        (413, "image_too_large", PayloadTooLargeError),
    ],
)
def test_known_provider_error_classifications_remain_available(
    model: str, status: int, error_type: str, error_class: type[ProviderError]
) -> None:
    settings = Settings.from_sources({"image_model": model, "review_model": model}, {})
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(status, json={"error": {"type": error_type}})
    )
    with (
        AgentBridgeClient(settings, transport=transport) as client,
        pytest.raises(error_class) as raised,
    ):
        if client._metered:
            client._metered._request("POST", "/images", json_body={}, retries=1)
        else:
            client._request("POST", "/images", json_body={})
    assert raised.value.status_code == status
    assert _error_name(raised.value) == error_type


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
                json={"data": [{"id": "codex/gpt-6.1-sol", "object": "model"}]},
            )
        body = json.loads(request.content)
        assert body["model"] == "codex/gpt-6.1-sol"
        if path == "/api/v1/chat/completions":
            assert body["reasoning_effort"] == "high"
        calls.append((path, body))
        if path == "/api/v1/images":
            return httpx.Response(
                200,
                json={
                    "data": [{"b64_json": base64.b64encode(generated).decode("ascii")}],
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

    with AgentBridgeClient(_settings(), transport=httpx.MockTransport(handler)) as client:
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
        quality_verdict = client.review_quality(candidate, view_name="full page")

    assert candidate.size == (16, 20)
    assert verdict.accepted is True
    assert quality_verdict.accepted is True
    assert projection.billing_mode == "codex_subscription"
    assert projection.backend_version == "0.1.9"
    assert projection.one_pass.cost_usd is None
    assert projection.one_pass.generations == 2
    assert projection.one_pass.reviews == 12
    assert calls[0][0] == "/api/v1/images"
    assert calls[0][1]["store"] is False
    assert calls[0][1]["n"] == 1
    assert calls[1][0] == "/api/v1/chat/completions"
    assert calls[1][1]["store"] is False
    assert calls[1][1]["response_format"]["type"] == "json_schema"
    quality_content = calls[2][1]["messages"][1]["content"]
    assert sum(item.get("type") == "image_url" for item in quality_content) == 1
    assert all(item.get("text") != "ORIGINAL:" for item in quality_content)
    assert client.costs.prompt_tokens == 11
    assert client.costs.completion_tokens == 4
    assert client.costs.total_tokens == 15


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

    with (
        AgentBridgeClient(_settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(GlobalProviderError, match="required authenticated"),
    ):
        client.preflight()


def test_agentbridge_generation_normalizes_invalid_image_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"b64_json": "not-base64"}]})

    with (
        AgentBridgeClient(_settings(), transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ProviderError, match="invalid image"),
    ):
        client.generate(
            Image.new("RGB", (8, 8), "white"),
            "clean",
            max_edge=4096,
        )
