from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from PIL import Image

from paperclean.config import Settings
from paperclean.discovery import output_paths
from paperclean.models import Discrepancy, ReviewVerdict
from paperclean.openrouter import CostTracker
from paperclean.pipeline import GENERATION_PROMPT, clean_image, report_has_fallback, report_summary
from paperclean.provenance import extract_png
from paperclean.validation import DeterministicResult


class FakeClient:
    def __init__(self, *, review_passes: bool = True) -> None:
        self.costs = CostTracker(None)
        self.review_passes = review_passes
        self.review_calls = 0

    def generate(self, source: Image.Image, _prompt: str, *, max_edge: int) -> Image.Image:
        assert max_edge > 0
        return source.copy()

    def review(
        self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
    ) -> ReviewVerdict:
        assert view_name
        self.review_calls += 1
        return ReviewVerdict(
            content_match=self.review_passes,
            scanner_quality=self.review_passes,
        )


def _write_png(path: Path) -> None:
    image = Image.new("RGB", (300, 400), "white")
    for x in range(40, 260):
        image.putpixel((x, 180), (0, 0, 0))
    image.save(path, format="PNG")


def test_generation_prompt_prioritizes_footer_sharpness_and_global_alignment() -> None:
    assert "every outer edge" in GENERATION_PROMPT
    assert "straight level baselines" in GENERATION_PROMPT
    assert all(word in GENERATION_PROMPT for word in ("blurred", "ghosted", "doubled"))
    assert "single coordinate system" in GENERATION_PROMPT


def test_clean_image_accepts_only_after_five_reviews_when_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    client = FakeClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", review_enabled=True),
        client,
        force=False,
    )  # type: ignore[arg-type]
    assert report.pages[0].status == "model_generated_clean"
    assert report.schema_version == 2
    assert report.review_enabled is True
    assert report.review_model == "openai/gpt-5.6-sol"
    assert client.review_calls == 5
    embedded = extract_png(paths.output.read_bytes())
    assert embedded is not None
    assert embedded["payload"]["pages"][0]["status"] == "model_generated_clean"
    sidecar = json.loads(paths.report.read_text())
    assert sidecar["output_sha256"] == report.output_sha256


def test_clean_image_default_skips_review_and_marks_provenance(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    client = FakeClient()
    paths = output_paths(source)
    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]

    assert client.review_calls == 0
    assert report.schema_version == 2
    assert report.review_enabled is False
    assert report.review_model is None
    assert report.pages[0].status == "model_generated_unreviewed"
    embedded = extract_png(paths.output.read_bytes())
    assert embedded is not None
    assert embedded["payload"]["schema_version"] == 2
    assert embedded["payload"]["review_enabled"] is False
    assert embedded["payload"]["models"]["review"] is None
    assert embedded["payload"]["pages"][0]["status"] == "model_generated_unreviewed"
    sidecar = json.loads(paths.report.read_text())
    assert sidecar["review_enabled"] is False
    assert sidecar["review_model"] is None
    assert report_has_fallback(report) is False
    assert json.loads(report_summary(report)) == {
        "output": str(paths.output),
        "generated_pages": 1,
        "fallback_pages": 0,
    }


def test_low_resolution_generation_is_upscaled_to_source_resolution_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    observed: dict[str, object] = {}

    def validate(source_image: Image.Image, candidate: Image.Image, **kwargs: object) -> object:
        observed["source_size"] = source_image.size
        observed["candidate_size"] = candidate.size
        observed["effective_dpi"] = kwargs["effective_dpi"]
        return DeterministicResult(True, [])

    monkeypatch.setattr("paperclean.pipeline.validate_candidate", validate)
    client = FakeClient()
    monkeypatch.setattr(
        client,
        "generate",
        lambda *_args, **_kwargs: Image.new("RGB", (75, 100), "white"),
    )
    paths = output_paths(source)

    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]

    attempt = report.pages[0].attempts[0]
    assert (attempt.generated_width, attempt.generated_height) == (75, 100)
    assert attempt.effective_dpi == 300
    assert observed == {
        "source_size": (300, 400),
        "candidate_size": (300, 400),
        "effective_dpi": 300.0,
    }


def test_clean_image_falls_back_and_preserves_source_idat(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    original = source.read_bytes()
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    client = FakeClient(review_passes=False)
    settings = Settings(api_key="fake", review_enabled=True, max_attempts=1, max_cost_usd=None)
    paths = output_paths(source)
    report = clean_image(paths, settings, client, force=False)  # type: ignore[arg-type]
    assert report.pages[0].status == "original_fallback"
    assert paths.output.read_bytes().startswith(original[:8])
    assert extract_png(paths.output.read_bytes()) is not None
    assert Decimal(str(report.cost_usd)) == Decimal("0.0")


def test_review_rejection_can_be_repaired_and_rechecked(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.rescue_edge_text",
        lambda _source, candidate, **_kwargs: candidate,
    )
    repaired = {"called": False}

    def repair(
        _source: Image.Image, candidate: Image.Image, *_args: object, **_kwargs: object
    ) -> Image.Image:
        repaired["called"] = True
        return candidate

    monkeypatch.setattr("paperclean.pipeline.repair_region", repair)

    class RepairClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls == 1:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[Discrepancy("changed_text", "high", (0.4, 0.4, 0.6, 0.5))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = RepairClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", review_enabled=True),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert repaired["called"] is True
    assert client.review_calls == 6
    assert report.pages[0].status == "model_generated_clean"
