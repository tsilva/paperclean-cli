from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from paperclean.models import Discrepancy
from paperclean.restoration import (
    REGIONAL_REPAIR_PROMPT,
    best_repair_region,
    clear_page_border,
    registered_review_pairs,
    repair_region,
    rescue_colored_marks,
    restore_source_regions,
)


def test_regional_prompt_requires_crisp_aligned_microprint() -> None:
    assert "crisp high-contrast glyph" in REGIONAL_REPAIR_PROMPT
    assert "Never duplicate" in REGIONAL_REPAIR_PROMPT


def test_clear_page_border_removes_outer_sliver_without_touching_content() -> None:
    image = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 30, 2), fill="black")
    draw.rectangle((50, 50, 80, 60), fill="black")

    result = clear_page_border(image)

    assert result.getpixel((10, 1)) == (255, 255, 255)
    assert result.getpixel((60, 55)) == (0, 0, 0)


def test_source_region_restore_preserves_mark_without_unrelated_dirt(monkeypatch) -> None:
    source = Image.new("RGB", (200, 300), (235, 225, 210))
    draw = ImageDraw.Draw(source)
    draw.ellipse((5, 90, 25, 110), fill="black")
    draw.line((120, 160, 150, 190), fill=(40, 40, 180), width=3)
    candidate = Image.new("RGB", source.size, "white")
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: np.eye(3),
    )

    result = restore_source_regions(source, candidate, [(0.55, 0.5, 0.8, 0.7)])

    assert result.getpixel((135, 175)) != (255, 255, 255)
    assert result.getpixel((15, 100)) == (255, 255, 255)


def test_colored_mark_rescue_keeps_signature_but_not_small_print(monkeypatch) -> None:
    source = Image.new("RGB", (400, 600), (240, 230, 215))
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, 399, 10), fill=(80, 55, 25))
    draw.text((30, 30), "BLUE HEADER", fill=(30, 40, 150))
    draw.line((230, 300, 300, 370), fill=(30, 30, 170), width=8)
    draw.ellipse((240, 290, 275, 345), outline=(30, 30, 170), width=7)
    candidate = Image.new("RGB", source.size, "white")
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: np.eye(3),
    )

    result = rescue_colored_marks(source, candidate)

    assert result.getpixel((260, 330)) != (255, 255, 255)
    assert result.getpixel((35, 35)) == (255, 255, 255)
    assert result.getpixel((100, 5)) == (255, 255, 255)


def test_best_repair_region_prefers_the_most_severe_content_issue() -> None:
    discrepancies = [
        Discrepancy("changed_text", "medium", (0.4, 0.4, 0.5, 0.45)),
        Discrepancy("scanner_quality", "critical", (0.0, 0.0, 1.0, 1.0)),
        Discrepancy("missing_text", "high", (0.6, 0.6, 0.7, 0.65)),
    ]

    assert best_repair_region(discrepancies) == (0.575, 0.575, 0.725, 0.675)


def test_registered_review_pairs_keeps_exact_full_page_and_aligns_regions(
    monkeypatch,
) -> None:
    source = Image.new("RGB", (100, 100), "white")
    source.putpixel((20, 20), (0, 0, 0))
    candidate = Image.new("RGB", (100, 100), "white")
    candidate.putpixel((30, 20), (0, 0, 0))
    # Candidate x=30 maps to source x=20, so source must shift right by 10.
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: np.array(
            [[1.0, 0.0, -0.1], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )

    pairs = registered_review_pairs(source, candidate)

    assert len(pairs) == 5
    assert pairs[0][0].tobytes() == source.tobytes()
    assert pairs[0][1].tobytes() == candidate.tobytes()
    assert pairs[1][0].getpixel((30, 20)) == (0, 0, 0)
    assert pairs[1][1].getpixel((30, 20)) == (0, 0, 0)


def test_regional_repair_splices_generated_crop(monkeypatch) -> None:
    source = Image.new("RGB", (200, 300), "white")
    ImageDraw.Draw(source).rectangle((80, 130, 120, 150), fill="black")
    candidate = Image.new("RGB", source.size, "white")
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: np.eye(3),
    )

    class Generator:
        calls = 0

        def generate(self, crop: Image.Image, prompt: str, *, max_edge: int) -> Image.Image:
            self.calls += 1
            assert prompt
            assert max_edge == 4096
            return crop.copy()

    client = Generator()
    result = repair_region(
        source,
        candidate,
        (0.35, 0.38, 0.65, 0.55),
        client=client,
        max_edge=4096,
    )

    assert client.calls == 1
    assert result.getbbox() == candidate.getbbox()
    assert result.getpixel((100, 140)) != (255, 255, 255)
