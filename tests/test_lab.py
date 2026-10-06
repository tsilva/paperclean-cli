from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from paperclean.imaging import decode_bytes, decode_data_url
from paperclean.lab import (
    Handler,
    Lab,
    LabServer,
    draw_composite,
    validate_box,
    validate_review,
)


def png(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (120, 180), color).save(buffer, format="PNG")
    return buffer.getvalue()


def review() -> dict[str, Any]:
    return {
        "summary": "One possible unsupported graphic.",
        "issues": [
            {
                "label": "Unsupported graphic",
                "category": "hallucinated_content",
                "severity": "high",
                "description": "An obscured graphic was completed.",
                "source_evidence": "The corresponding source portion is covered.",
                "restored_box": {"x0": 0.65, "y0": 0.1, "x1": 0.9, "y1": 0.3},
                "source_box": {"x0": 0.7, "y0": 0.1, "x1": 0.9, "y1": 0.35},
            }
        ],
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"x0": -0.1, "y0": 0.1, "x1": 0.2, "y1": 0.3},
        {"x0": 0.5, "y0": 0.1, "x1": 0.2, "y1": 0.3},
        {"x0": 0.1, "y0": 0.1, "x1": float("nan"), "y1": 0.3},
        {"x0": True, "y0": 0.1, "x1": 0.2, "y1": 0.3},
        {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 1.2},
    ],
)
def test_invalid_coordinates_are_rejected_instead_of_silently_clamped(bad: Any) -> None:
    with pytest.raises(ValueError):
        validate_box(bad)


def test_composite_preserves_pixels_outside_annotation_region() -> None:
    candidate = Image.fromarray(
        np.random.default_rng(5).integers(0, 255, (500, 400, 3), dtype=np.uint8)
    )
    issues = validate_review(review())["issues"]
    composite = draw_composite(candidate, issues)
    original = np.asarray(candidate)
    assert np.array_equal(np.asarray(composite)[250:, :, :], original[250:, :, :])
    assert not np.array_equal(np.asarray(composite), original)
    assert composite.size == candidate.size


@pytest.mark.parametrize("existing", [False, True])
def test_experiment_uses_native_generation_then_exact_sol_high_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    lab = Lab(tmp_path, "http://127.0.0.1:8082/api/v1")
    monkeypatch.setattr(lab, "capabilities", lambda: {"ready": True})
    calls: list[tuple[str, dict[str, Any]]] = []

    def request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((path, payload))
        if path == "/images":
            return {"data": [{"b64_json": base64.b64encode(png("blue")).decode()}]}
        return {
            "choices": [{"message": {"content": json.dumps(review())}}],
            "usage": {"total_tokens": 30},
        }

    monkeypatch.setattr(lab, "request", request)
    document = lab.import_document("scan.png", png())
    run = lab.start_run(
        {
            "approved": True,
            "document_id": document["id"],
            "page": 1,
            "mode": "review_existing" if existing else "restore_and_review",
            **({"candidate": base64.b64encode(png("blue")).decode()} if existing else {}),
        }
    )
    lab.executor.shutdown(wait=True)
    saved = lab.state()["runs"][0]
    assert saved["status"] == "complete"
    assert [path for path, _ in calls] == (
        ["/chat/completions"] if existing else ["/images", "/chat/completions"]
    )
    body = calls[-1][1]
    assert body["model"] == "codex/gpt-6.1-sol"
    assert body["reasoning_effort"] == "high"
    images = [part for part in body["messages"][1]["content"] if part["type"] == "image_url"]
    assert len(images) == 2
    assert decode_bytes(decode_data_url(images[0]["image_url"]["url"])).getpixel((0, 0)) == (
        255,
        255,
        255,
    )
    assert decode_bytes(decode_data_url(images[1]["image_url"]["url"])).getpixel((0, 0)) == (
        0,
        0,
        255,
    )
    assert body["response_format"]["json_schema"]["strict"] is True
    assert saved["production_accepted"] is False
    assert saved["usd_cost"] is None
    assert (tmp_path / "runs" / run["id"] / "composite.png").is_file()
    lab.judgments(
        run["id"], {"judgments": {"1": "false_alarm"}, "notes": "Needs closer inspection"}
    )
    assert lab.state()["runs"][0]["judgments"]["1"] == "false_alarm"


def test_no_generation_without_explicit_ui_confirmation(tmp_path: Path) -> None:
    lab = Lab(tmp_path, "http://127.0.0.1:8082/api/v1")
    with pytest.raises(ValueError, match="Confirm transmission"):
        lab.start_run({"approved": False})
    lab.executor.shutdown()


def test_rejected_review_keeps_candidate_and_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab = Lab(tmp_path, "http://127.0.0.1:8082/api/v1")
    monkeypatch.setattr(lab, "capabilities", lambda: {"ready": True})
    calls = []

    def request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(path)
        value = review()
        value["issues"][0]["restored_box"]["x1"] = 5
        return {"choices": [{"message": {"content": json.dumps(value)}}]}

    monkeypatch.setattr(lab, "request", request)
    document = lab.import_document("scan.png", png())
    run = lab.start_run(
        {
            "approved": True,
            "document_id": document["id"],
            "page": 1,
            "mode": "review_existing",
            "candidate": base64.b64encode(png()).decode(),
        }
    )
    lab.executor.shutdown(wait=True)
    saved = lab.state()["runs"][0]
    assert saved["status"] == "failed"
    assert saved["restored_url"] is not None
    assert saved["composite_url"] is None
    assert calls == ["/chat/completions"]
    assert (tmp_path / "runs" / run["id"] / "restored.png").is_file()


def test_http_blocks_cross_origin_mutations_and_path_traversal(tmp_path: Path) -> None:
    lab = Lab(tmp_path, "http://127.0.0.1:8082/api/v1")
    server = LabServer(("127.0.0.1", 0), Handler)
    server.lab = lab
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}") as client:
            assert client.get("/").status_code == 200
            assert client.get("/api/state", headers={"Host": "attacker.example"}).status_code == 403
            assert (
                client.post(
                    "/api/documents", json={}, headers={"Origin": "https://attacker.example"}
                ).status_code
                == 403
            )
            assert client.get("/files/runs/" + "a" * 32 + "/outside.txt").status_code == 404
            assert client.get("/files/%2e%2e/pyproject.toml").status_code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        lab.executor.shutdown()


def test_restart_marks_incomplete_runs_as_interrupted(tmp_path: Path) -> None:
    directory = tmp_path / "runs" / ("a" * 32)
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({"status": "reviewing"}))
    lab = Lab(tmp_path, "http://127.0.0.1:8082/api/v1")
    assert json.loads((directory / "run.json").read_text())["status"] == "interrupted"
    lab.executor.shutdown()
