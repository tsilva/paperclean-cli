from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from paperclean.config import Settings
from paperclean.discovery import output_paths
from paperclean.errors import ProviderError, ReviewerResponseError
from paperclean.models import Discrepancy, PageGeometry, ReviewVerdict
from paperclean.openrouter import CostTracker
from paperclean.pipeline import GENERATION_PROMPT, clean_image, report_has_fallback, report_summary
from paperclean.prompting import PUNCH_HOLE_REPAIR_PROMPT
from paperclean.provenance import extract_png
from paperclean.restoration import PagePlane
from paperclean.validation import DeterministicResult


class FakeClient:
    def __init__(self, *, review_passes: bool = True) -> None:
        self.costs = CostTracker(None)
        self.review_passes = review_passes
        self.review_calls = 0

    def generate(self, source: Image.Image, _prompt: str, *, max_edge: int) -> Image.Image:
        assert max_edge > 0
        return source.copy()

    def locate_page(self, _source: Image.Image) -> PageGeometry | None:
        return None

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


def test_clean_image_accepts_only_after_five_model_verifications(
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
        Settings(api_key="fake"),
        client,
        force=False,
    )  # type: ignore[arg-type]
    assert report.pages[0].status == "model_generated_clean"
    assert report.schema_version == 6
    assert report.backend == "openrouter"
    assert report.billing_mode == "openrouter_usd"
    assert report.verification_model == "openai/gpt-5.6-sol"
    assert report.verification_strategy == "full-page-plus-four-registered-regions"
    assert client.review_calls == 5
    embedded = extract_png(paths.output.read_bytes())
    assert embedded is not None
    assert embedded["payload"]["schema_version"] == 6
    assert embedded["payload"]["models"]["verification"] == "openai/gpt-5.6-sol"
    assert embedded["payload"]["verification"]["strategy"] == report.verification_strategy
    assert embedded["payload"]["pages"][0]["status"] == "model_generated_clean"
    sidecar = json.loads(paths.report.read_text())
    assert sidecar["output_sha256"] == report.output_sha256
    assert report_has_fallback(report) is False
    assert json.loads(report_summary(report)) == {
        "output": str(paths.output),
        "generated_pages": 1,
        "fallback_pages": 0,
    }


def test_photographed_page_is_rectified_then_model_recreated_and_verified(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class PhotoClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls = 0
            self.locate_calls = 0

        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            self.locate_calls += 1
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

        def generate(self, source: Image.Image, _prompt: str, *, max_edge: int) -> Image.Image:
            self.generate_calls += 1
            assert max_edge > 0
            return source.copy()

    client = PhotoClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    page = report.pages[0]
    assert page.status == "model_assisted_clean"
    assert client.locate_calls == 1
    assert client.generate_calls == 1
    assert client.review_calls == 5
    assert page.attempts[0].generated_width == 300
    assert page.attempts[0].generated_height == 400


def test_photographed_page_semantic_review_arbitrates_fold_foreground_loss(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(False, ["large_foreground_loss"]),
    )

    class FoldedPhotoClient(FakeClient):
        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

    client = FoldedPhotoClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    assert report.pages[0].status == "model_assisted_clean"
    assert client.review_calls == 5
    assert report.pages[0].attempts[0].local_issues == ["large_foreground_loss"]


@pytest.mark.parametrize(
    "issue",
    [
        "page_registration_failed",
        "candidate_canvas_mismatch",
        "large_candidate_only_foreground",
        "generated_resolution_below_minimum",
    ],
)
def test_photographed_page_keeps_non_fold_local_failures_as_hard_blocks(
    tmp_path: Path, monkeypatch, issue: str
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(False, [issue]),
    )

    class InvalidPhotoClient(FakeClient):
        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

    client = InvalidPhotoClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    assert report.pages[0].status == "original_fallback"
    assert client.review_calls == 0


def test_photographed_page_restores_changed_authored_region_before_retrying_generation(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    restored_regions: list[list[tuple[float, float, float, float]]] = []

    def restore(_source, candidate, regions, **_kwargs):
        restored_regions.append(list(regions))
        return candidate

    monkeypatch.setattr("paperclean.pipeline.replace_with_source_evidence_regions", restore)

    class ChangedStampClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls = 0

        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

        def generate(self, source: Image.Image, _prompt: str, *, max_edge: int) -> Image.Image:
            self.generate_calls += 1
            return source.copy()

        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[
                        Discrepancy("changed_stamp", "high", (0.55, 0.70, 0.80, 0.90))
                    ],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = ChangedStampClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=2),
        client,  # type: ignore[arg-type]
        force=False,
    )

    assert report.pages[0].status == "model_assisted_clean"
    assert client.generate_calls == 1
    assert client.review_calls == 11
    assert restored_regions == [[(0.55, 0.70, 0.80, 0.90)]]


def test_photographed_page_iteratively_restores_newly_exposed_source_regions(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    restored_regions: list[list[tuple[float, float, float, float]]] = []

    def restore(_source, candidate, regions, **_kwargs):
        restored_regions.append(list(regions))
        return candidate

    monkeypatch.setattr("paperclean.pipeline.replace_with_source_evidence_regions", restore)

    class IterativeClient(FakeClient):
        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                discrepancy = Discrepancy("missing_text", "high", (0.1, 0.1, 0.3, 0.2))
                return ReviewVerdict(False, True, [discrepancy])
            if self.review_calls <= 4:
                discrepancy = Discrepancy("changed_stamp", "high", (0.6, 0.7, 0.8, 0.85))
                return ReviewVerdict(False, True, [discrepancy])
            return ReviewVerdict(True, True)

    client = IterativeClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    assert report.pages[0].status == "model_assisted_clean"
    assert restored_regions == [
        [(0.1, 0.1, 0.3, 0.2), pytest.approx((0.33, 0.385, 0.44, 0.4675))]
    ]


def test_photographed_page_repairs_localized_quality_region_after_content_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    repairs: list[tuple[float, float, float, float]] = []

    def repair(_source, candidate, region, **_kwargs):
        repairs.append(region)
        return candidate

    monkeypatch.setattr("paperclean.pipeline.repair_region", repair)

    class QualityRepairClient(FakeClient):
        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 3:
                discrepancy = Discrepancy("scanner_quality", "high", (0.2, 0.2, 0.4, 0.35))
                return ReviewVerdict(True, False, [discrepancy])
            return ReviewVerdict(True, True)

    client = QualityRepairClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    assert report.pages[0].status == "model_assisted_clean"
    assert len(repairs) == 1
    assert repairs[0] == pytest.approx((0.15, 0.125, 0.45, 0.425))


def test_photographed_page_uses_verified_source_preserving_rectification_when_generation_fails(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class SourceFallbackClient(FakeClient):
        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 50:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[Discrepancy("invented_text", "high", (0.2, 0.2, 0.8, 0.8))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = SourceFallbackClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    page = report.pages[0]
    assert page.status == "source_preserving_clean"
    assert [attempt.strategy for attempt in page.attempts] == [
        "model_assisted_source_cleanup",
        "source_preserving_cleanup",
    ]
    assert page.attempts[-1].accepted is True


def test_photographed_source_fallback_repairs_localized_quality_region(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "photo.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.detect_page_plane",
        lambda _source: PagePlane(
            corners=np.asarray(((0, 0), (299, 0), (299, 399), (0, 399)), dtype=np.float32),
            area_fraction=0.8,
            confidence=0.95,
        ),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    repairs: list[tuple[float, float, float, float]] = []

    def repair(_source, candidate, region, **_kwargs):
        repairs.append(region)
        return candidate

    monkeypatch.setattr("paperclean.pipeline.repair_region", repair)

    class QualityFallbackClient(FakeClient):
        def locate_page(self, _source: Image.Image) -> PageGeometry | None:
            return PageGeometry(
                corners=((0.0, 0.0), (0.997, 0.0), (0.997, 0.997), (0.0, 0.997)),
                content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
                occlusions=(),
                confidence=0.95,
            )

        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 50:
                return ReviewVerdict(
                    False,
                    True,
                    [Discrepancy("invented_text", "high", (0.2, 0.2, 0.8, 0.8))],
                )
            if self.review_calls <= 53:
                return ReviewVerdict(
                    True,
                    False,
                    [Discrepancy("scanner_quality", "high", (0.2, 0.2, 0.4, 0.35))],
                )
            return ReviewVerdict(True, True)

    client = QualityFallbackClient()
    report = clean_image(
        output_paths(source),
        Settings(api_key="fake", max_attempts=1),
        client,  # type: ignore[arg-type]
        force=False,
    )

    assert report.pages[0].status == "source_preserving_clean"
    assert len(repairs) == 1
    assert repairs[0] == pytest.approx((0.15, 0.125, 0.45, 0.425))


def test_clean_image_records_agentbridge_subscription_provenance(
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
    settings = Settings(
        api_key="",
        backend="agentbridge",
        base_url="http://127.0.0.1:8082/api/v1",
        image_model="codex/gpt-5.6-sol",
        review_model="codex/gpt-5.6-sol",
    )

    report = clean_image(paths, settings, client, force=False)  # type: ignore[arg-type]

    assert report.backend == "agentbridge"
    assert report.billing_mode == "codex_subscription"
    assert report.cost_usd is None
    sidecar = json.loads(paths.report.read_text())
    assert sidecar["backend"] == "agentbridge"
    assert sidecar["billing_mode"] == "codex_subscription"
    assert sidecar["cost_usd"] is None


def test_agentbridge_ordinary_scan_verifies_source_cleanup_before_generation(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.authored_punch_hole_regions",
        lambda _source: [],
    )

    class SourceFirstClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.generate_calls = 0

        def generate(
            self, source: Image.Image, _prompt: str, *, max_edge: int
        ) -> Image.Image:
            self.generate_calls += 1
            return source.copy()

    client = SourceFirstClient()
    settings = Settings(
        api_key="",
        backend="agentbridge",
        base_url="http://127.0.0.1:8082/api/v1",
        image_model="codex/gpt-5.6-sol",
        review_model="codex/gpt-5.6-sol",
    )

    report = clean_image(output_paths(source), settings, client, force=False)  # type: ignore[arg-type]

    assert report.pages[0].status == "source_preserving_clean"
    assert report.pages[0].attempts[0].strategy == "source_preserving_cleanup"
    assert client.generate_calls == 0
    assert client.review_calls == 5


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
    settings = Settings(api_key="fake", max_attempts=1, max_cost_usd=None)
    paths = output_paths(source)
    report = clean_image(paths, settings, client, force=False)  # type: ignore[arg-type]
    assert report.pages[0].status == "original_fallback"
    assert paths.output.read_bytes().startswith(original[:8])
    assert extract_png(paths.output.read_bytes()) is not None
    assert Decimal(str(report.cost_usd)) == Decimal("0.0")


def test_invalid_verification_never_accepts_a_generated_page(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class InvalidVerifier(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            raise ReviewerResponseError("invalid structured verdict")

    client = InvalidVerifier()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 4
    assert report.pages[0].status == "original_fallback"
    assert report.pages[0].fallback_reason == "provider_or_review_error"
    assert [attempt.strategy for attempt in report.pages[0].attempts] == [
        "model_generation",
        "source_preserving_cleanup",
    ]


def test_source_preserving_cleanup_recovers_after_transient_reviewer_error(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class TransientReviewerErrorClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                raise ReviewerResponseError("transient invalid structured verdict")
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = TransientReviewerErrorClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    page = report.pages[0]
    assert client.review_calls == 7
    assert page.status == "source_preserving_clean"
    assert [attempt.strategy for attempt in page.attempts] == [
        "model_generation",
        "source_preserving_cleanup",
    ]
    assert page.attempts[-1].accepted is True


def test_page_review_retries_one_transient_timeout(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class TimeoutThenPassClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls == 1:
                raise ProviderError("transient timeout", error_type="timeout_error")
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = TimeoutThenPassClient()
    paths = output_paths(source)
    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]

    assert client.review_calls == 6
    assert report.pages[0].status == "model_generated_clean"


def test_transient_quality_only_rejection_is_confirmed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class FlakyQualityClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls == 1:
                return ReviewVerdict(
                    content_match=True,
                    scanner_quality=False,
                    discrepancies=[Discrepancy("scanner_quality", "high", (0.0, 0.0, 1.0, 1.0))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = FlakyQualityClient()
    paths = output_paths(source)
    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]

    assert client.review_calls == 6
    assert report.pages[0].status == "model_generated_clean"


def test_confirmed_quality_only_rejection_fails_closed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class PoorQualityClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            return ReviewVerdict(content_match=True, scanner_quality=False)

    client = PoorQualityClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 4
    assert report.pages[0].status == "original_fallback"
    assert report.pages[0].attempts[0].verification_categories == ["scanner_quality"]
    assert report.pages[0].attempts[1].strategy == "source_preserving_cleanup"


def test_source_preserving_cleanup_recovers_after_model_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class SourceRecoveryClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = SourceRecoveryClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    page = report.pages[0]
    assert client.review_calls == 7
    assert page.status == "source_preserving_clean"
    assert [attempt.strategy for attempt in page.attempts] == [
        "model_generation",
        "source_preserving_cleanup",
    ]
    assert page.attempts[-1].accepted is True
    assert report_has_fallback(report) is False


def test_high_confidence_authored_hole_repair_is_published(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.authored_punch_hole_regions",
        lambda _source: [(0.0, 0.3, 0.3, 0.4)],
    )
    repair_prompts: list[str] = []

    def repair(_source, candidate, *_args, **kwargs):
        repair_prompts.append(kwargs["prompt"])
        return candidate

    monkeypatch.setattr("paperclean.pipeline.repair_region", repair)

    class VerifiedRepairClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = VerifiedRepairClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    page = report.pages[0]
    assert client.review_calls == 7
    assert page.status == "model_assisted_clean"
    assert [attempt.strategy for attempt in page.attempts] == [
        "model_generation",
        "model_assisted_source_cleanup",
    ]
    assert repair_prompts == [PUNCH_HOLE_REPAIR_PROMPT]
    assert page.attempts[-1].accepted is True


@pytest.mark.parametrize("category", ["changed_text", "unresolved_content", "other_content"])
def test_uncertain_authored_hole_repair_keeps_the_source_hole(
    tmp_path: Path, monkeypatch, category: str
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.authored_punch_hole_regions",
        lambda _source: [(0.0, 0.3, 0.3, 0.4)],
    )
    monkeypatch.setattr(
        "paperclean.pipeline.repair_region",
        lambda _source, candidate, *_args, **_kwargs: candidate,
    )

    class UncertainRepairClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            if self.review_calls <= 4:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[Discrepancy(category, "high", (0.0, 0.0, 0.3, 0.4))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = UncertainRepairClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    page = report.pages[0]
    assert client.review_calls == 9
    assert page.status == "source_preserving_clean"
    assert [attempt.strategy for attempt in page.attempts] == [
        "model_generation",
        "model_assisted_source_cleanup",
        "source_preserving_cleanup",
    ]
    assert page.attempts[-2].accepted is False
    assert page.attempts[-1].accepted is True


def test_source_cleanup_records_confirmed_quality_limitations(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class LowQualityClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            if self.review_calls <= 4:
                return ReviewVerdict(
                    content_match=True,
                    scanner_quality=False,
                    discrepancies=[Discrepancy("scanner_quality", "high", (0.0, 0.0, 1.0, 1.0))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = LowQualityClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    page = report.pages[0]
    assert client.review_calls == 8
    assert page.status == "source_preserving_clean"
    assert page.attempts[-1].accepted is True
    assert page.attempts[-1].verification_categories == ["scanner_quality"]


def test_source_cleanup_tolerates_only_expected_layout_rectification(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class RectifiedLayoutClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            return ReviewVerdict(
                content_match=False,
                scanner_quality=True,
                discrepancies=[Discrepancy("changed_layout", "medium", (0.0, 0.0, 1.0, 1.0))],
            )

    client = RectifiedLayoutClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 12
    assert report.pages[0].status == "source_preserving_clean"
    assert report.pages[0].attempts[-1].verification_categories == ["changed_layout"]


def test_source_cleanup_tolerates_changed_diagram_only_for_preserved_photographic_regions(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    monkeypatch.setattr(
        "paperclean.pipeline.has_preserved_photographic_regions",
        lambda _source: True,
    )

    class PreservedPhotoClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            return ReviewVerdict(
                content_match=False,
                scanner_quality=True,
                discrepancies=[Discrepancy("changed_diagram", "high", (0.1, 0.1, 0.9, 0.9))],
            )

    client = PreservedPhotoClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 12
    assert report.pages[0].status == "source_preserving_clean"
    assert report.pages[0].attempts[-1].verification_categories == ["changed_diagram"]


@pytest.mark.parametrize("category", ["other_content", "unresolved_content"])
def test_source_cleanup_records_confirmed_non_specific_alerts(
    tmp_path: Path, monkeypatch, category: str
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class NonSpecificAlertClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            return ReviewVerdict(
                content_match=False,
                scanner_quality=True,
                discrepancies=[Discrepancy(category, "high", (0.0, 0.0, 1.0, 1.0))],
            )

    client = NonSpecificAlertClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 12
    assert report.pages[0].status == "source_preserving_clean"
    assert report.pages[0].attempts[-1].verification_categories == [category]


def test_source_cleanup_confirms_transient_content_rejection(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class TransientContentClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            if self.review_calls == 3:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[Discrepancy("changed_table", "high", (0.1, 0.1, 0.9, 0.9))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = TransientContentClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 8
    assert report.pages[0].status == "source_preserving_clean"
    assert report.pages[0].attempts[-1].accepted is True


def test_source_cleanup_records_each_boolean_only_rejection(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )

    class RegionalBooleanClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            if self.review_calls <= 4:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[
                        Discrepancy("changed_layout", "medium", (0.0, 0.0, 1.0, 1.0))
                    ],
                )
            if self.review_calls <= 6:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = RegionalBooleanClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert client.review_calls == 6
    assert report.pages[0].status == "original_fallback"
    assert report.pages[0].attempts[-1].verification_categories == [
        "changed_layout",
        "scanner_quality",
    ]


def test_source_cleanup_restores_flagged_source_evidence_and_rechecks(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    restored_regions: list[list[tuple[float, float, float, float]]] = []

    def restore(
        _source: Image.Image,
        candidate: Image.Image,
        regions: list[tuple[float, float, float, float]],
    ) -> Image.Image:
        restored_regions.append(regions)
        return candidate

    monkeypatch.setattr("paperclean.pipeline.restore_source_evidence_regions", restore)

    class MissingEvidenceClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls <= 2:
                return ReviewVerdict(content_match=True, scanner_quality=False)
            if self.review_calls <= 4:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[
                        Discrepancy("changed_text", "high", (0.2, 0.3, 0.8, 0.4))
                    ],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = MissingEvidenceClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake", max_attempts=1),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert restored_regions == [[(0.2, 0.3, 0.8, 0.4)]]
    assert client.review_calls == 9
    assert report.pages[0].status == "source_preserving_clean"
    assert report.pages[0].attempts[-1].accepted is True


def test_review_rejection_can_be_repaired_and_rechecked(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
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
                    discrepancies=[Discrepancy("other_content", "high", (0.4, 0.4, 0.6, 0.5))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = RepairClient()
    paths = output_paths(source)
    report = clean_image(
        paths,
        Settings(api_key="fake"),
        client,
        force=False,
    )  # type: ignore[arg-type]

    assert repaired["called"] is True
    assert client.review_calls == 6
    assert report.pages[0].status == "model_generated_clean"


@pytest.mark.parametrize("category", ["missing_text", "changed_layout"])
def test_edge_verification_rejection_restores_source_and_rechecks(
    tmp_path: Path, monkeypatch, category: str
) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    restored_regions: list[list[tuple[float, float, float, float]]] = []

    def restore(
        _source: Image.Image,
        candidate: Image.Image,
        regions: list[tuple[float, float, float, float]],
    ) -> Image.Image:
        restored_regions.append(regions)
        return candidate

    monkeypatch.setattr("paperclean.pipeline.restore_source_regions", restore)

    class EdgeClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls == 1:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[Discrepancy(category, "high", (0.2, 0.96, 0.8, 0.99))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = EdgeClient()
    paths = output_paths(source)
    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]

    assert restored_regions == [[(0.0, 0.94, 1.0, 1.0)]]
    assert client.review_calls == 6
    assert report.pages[0].status == "model_generated_clean"


def test_interior_content_rejection_restores_registered_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "scan.png"
    _write_png(source)
    monkeypatch.setattr(
        "paperclean.pipeline.validate_candidate",
        lambda *_args, **_kwargs: DeterministicResult(True, []),
    )
    restored_regions: list[list[tuple[float, float, float, float]]] = []

    def restore(
        _source: Image.Image,
        candidate: Image.Image,
        regions: list[tuple[float, float, float, float]],
    ) -> Image.Image:
        restored_regions.append(regions)
        return candidate

    monkeypatch.setattr("paperclean.pipeline.restore_source_regions", restore)

    class TableClient(FakeClient):
        def review(
            self, _source: Image.Image, _candidate: Image.Image, *, view_name: str
        ) -> ReviewVerdict:
            self.review_calls += 1
            if self.review_calls == 1:
                return ReviewVerdict(
                    content_match=False,
                    scanner_quality=True,
                    discrepancies=[Discrepancy("changed_table", "high", (0.2, 0.3, 0.8, 0.7))],
                )
            return ReviewVerdict(content_match=True, scanner_quality=True)

    client = TableClient()
    paths = output_paths(source)
    report = clean_image(paths, Settings(api_key="fake"), client, force=False)  # type: ignore[arg-type]

    assert restored_regions == [[(0.2, 0.3, 0.8, 0.7)]]
    assert client.review_calls == 6
    assert report.pages[0].status == "model_generated_clean"
