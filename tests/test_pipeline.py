from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from PIL import Image

from paperclean.config import Settings
from paperclean.discovery import output_paths
from paperclean.models import ReviewVerdict
from paperclean.openrouter import CostTracker
from paperclean.pipeline import clean_image
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


def test_clean_image_accepts_only_after_five_reviews(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    client = FakeClient()
    paths = output_paths(source)
    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]
    assert report.pages[0].status == "model_generated_clean"
    assert client.review_calls == 5
    embedded = extract_png(paths.output.read_bytes())
    assert embedded is not None
    assert embedded["payload"]["pages"][0]["status"] == "model_generated_clean"
    sidecar = json.loads(paths.report.read_text())
    assert sidecar["output_sha256"] == report.output_sha256


def test_clean_image_falls_back_and_preserves_source_idat(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    original = source.read_bytes()
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    client = FakeClient(review_passes=False)
    settings = Settings(api_key="fake", max_attempts=1, max_cost_usd=None)
    paths = output_paths(source)
    report = clean_image(paths, settings, client, force=False)  # type: ignore[arg-type]
    assert report.pages[0].status == "original_fallback"
    assert paths.output.read_bytes().startswith(original[:8])
    assert extract_png(paths.output.read_bytes()) is not None
    assert Decimal(str(report.cost_usd)) == Decimal("0.0")
