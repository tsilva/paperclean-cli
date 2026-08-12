from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from paperclean.models import Discrepancy
from paperclean.prompting import PUNCH_HOLE_REPAIR_PROMPT
from paperclean.restoration import (
    REGIONAL_REPAIR_PROMPT,
    _photographic_region_mask,
    _punch_hole_candidates,
    authored_punch_hole_regions,
    best_repair_region,
    clear_page_border,
    has_preserved_photographic_regions,
    registered_review_pairs,
    repair_region,
    rescue_colored_marks,
    restore_source_evidence_regions,
    restore_source_regions,
    source_preserving_cleanup,
)


def test_regional_prompt_requires_crisp_aligned_microprint() -> None:
    assert "crisp high-contrast glyph" in REGIONAL_REPAIR_PROMPT
    assert "Never duplicate" in REGIONAL_REPAIR_PROMPT


def test_punch_hole_prompt_requires_high_probability_reconstruction() -> None:
    assert "highly probable" in PUNCH_HOLE_REPAIR_PROMPT
    assert "not unambiguous" in PUNCH_HOLE_REPAIR_PROMPT
    assert "instead of guessing" in PUNCH_HOLE_REPAIR_PROMPT


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


def test_source_evidence_restore_recovers_only_flagged_faint_strokes(monkeypatch) -> None:
    source = Image.new("RGB", (200, 300), "white")
    draw = ImageDraw.Draw(source)
    draw.line((70, 100, 130, 100), fill=(244, 244, 244), width=2)
    draw.line((70, 240, 130, 240), fill=(244, 244, 244), width=2)
    candidate = Image.new("RGB", source.size, "white")
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: np.eye(3),
    )

    result = restore_source_evidence_regions(
        source,
        candidate,
        [(0.3, 0.25, 0.7, 0.4)],
    )

    assert result.getpixel((100, 100))[0] < 255
    assert result.getpixel((100, 240)) == (255, 255, 255)


def test_source_preserving_cleanup_removes_rails_and_punches_but_keeps_content() -> None:
    source = Image.new("RGB", (400, 600), (238, 225, 205))
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, 399, 7), fill="black")
    draw.rectangle((60, 20, 340, 35), fill="black")
    draw.ellipse((5, 95, 25, 115), fill="black")
    draw.ellipse((35, 145, 55, 165), outline="black", width=3)
    draw.rectangle((100, 200, 250, 215), fill="black")
    draw.line((180, 350, 260, 410), fill=(30, 30, 170), width=5)

    result = source_preserving_cleanup(source)

    assert result.getpixel((200, 3)) == (255, 255, 255)
    assert result.getpixel((200, 27)) != (255, 255, 255)
    assert result.getpixel((15, 105)) == (255, 255, 255)
    assert result.getpixel((45, 145)) != (255, 255, 255)
    assert result.getpixel((150, 207)) != (255, 255, 255)
    assert result.getpixel((220, 380)) != (255, 255, 255)
    assert result.getpixel((300, 300)) == (255, 255, 255)


def test_source_cleanup_keeps_a_punch_hole_that_touches_authored_ink() -> None:
    source = Image.new("RGB", (500, 700), (238, 225, 205))
    draw = ImageDraw.Draw(source)
    draw.ellipse((5, 190, 25, 210), fill="black")
    draw.rectangle((20, 193, 150, 207), fill="black")

    result = source_preserving_cleanup(source)

    assert result.getpixel((15, 200)) != (255, 255, 255)
    assert result.getpixel((80, 200)) != (255, 255, 255)
    regions = authored_punch_hole_regions(source)
    assert len(regions) == 1
    assert regions[0][0] < 0.01
    assert regions[0][2] > 0.25


def test_distance_transform_finds_hole_merged_into_text_line() -> None:
    pixels = np.full((800, 800, 3), 238, dtype=np.uint8)
    cv2.circle(pixels, (45, 300), 25, (0, 0, 0), thickness=-1)
    cv2.rectangle(pixels, (45, 292), (300, 308), (0, 0, 0), thickness=-1)
    source = Image.fromarray(pixels, "RGB")

    candidates = _punch_hole_candidates(pixels)

    assert any(
        abs(center_x - 45) <= 3
        and abs(center_y - 300) <= 3
        and touches_authored_ink
        for center_x, center_y, _radius, _padding, touches_authored_ink in candidates
    )
    assert authored_punch_hole_regions(source)


def test_source_cleanup_removes_a_blank_punch_hole_near_ten_percent_margin() -> None:
    source = Image.new("RGB", (400, 600), (238, 225, 205))
    draw = ImageDraw.Draw(source)
    draw.ellipse((16, 190, 36, 210), fill="black")

    result = source_preserving_cleanup(source)

    assert result.getpixel((26, 200)) == (255, 255, 255)
    assert authored_punch_hole_regions(source) == []


def test_fragmented_antialiased_hole_rim_is_not_authored_ink() -> None:
    pixels = np.full((800, 600, 3), 238, dtype=np.uint8)
    cv2.circle(pixels, (35, 300), 20, (0, 0, 0), thickness=-1)
    cv2.circle(pixels, (35, 300), 24, (155, 155, 155), thickness=1)

    candidates = _punch_hole_candidates(pixels)

    assert candidates
    assert all(touches_authored_ink is False for *_geometry, touches_authored_ink in candidates)


def test_source_cleanup_removes_punch_hole_halo() -> None:
    pixels = np.full((800, 600, 3), 238, dtype=np.uint8)
    cv2.circle(pixels, (35, 300), 20, (0, 0, 0), thickness=-1)
    cv2.circle(pixels, (35, 300), 25, (170, 170, 170), thickness=4)

    cleaned = np.asarray(source_preserving_cleanup(Image.fromarray(pixels, "RGB")))

    assert np.all(cleaned[270:331, 5:66] == 255)


def test_source_preserving_cleanup_globally_deskews_without_cropping() -> None:
    source = Image.new("RGB", (600, 800), (238, 225, 205))
    draw = ImageDraw.Draw(source)
    for y in (150, 300, 450, 600):
        draw.line((80, y, 520, y), fill="black", width=4)
    source = source.rotate(2.0, resample=Image.Resampling.BICUBIC, fillcolor="white")

    result = source_preserving_cleanup(source)
    gray = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2GRAY)
    lines = cv2.HoughLinesP(
        cv2.Canny(gray, 50, 150),
        1,
        np.pi / 1800,
        threshold=100,
        minLineLength=300,
        maxLineGap=20,
    )
    assert lines is not None
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]
    assert abs(float(np.median(angles))) < 0.3
    assert result.size == source.size


def test_source_preserving_cleanup_keeps_large_raster_panels_exact() -> None:
    source = Image.new("RGB", (800, 1000), (230, 215, 190))
    pixels = np.asarray(source).copy()
    y1, y2, x1, x2 = 180, 580, 40, 680
    yy, xx = np.mgrid[y1:y2, x1:x2]
    texture = ((xx * 7 + yy * 11) % 150).astype(np.uint8)
    pixels[y1:y2, x1:x2] = np.stack((texture, texture, texture), axis=2)
    cv2.circle(pixels, (60, 300), 12, (0, 0, 0), thickness=-1)
    source = Image.fromarray(pixels, "RGB")

    cleaned = np.asarray(source_preserving_cleanup(source))

    assert np.array_equal(cleaned[y1:y2, x1:x2], pixels[y1:y2, x1:x2])
    assert tuple(cleaned[50, 400]) == (255, 255, 255)
    assert has_preserved_photographic_regions(source) is True
    assert authored_punch_hole_regions(source) == []
    assert has_preserved_photographic_regions(Image.new("RGB", source.size, "white")) is False


def test_photographic_panel_survives_thin_noise_bridge_to_scan_border() -> None:
    pixels = np.full((1000, 800, 3), 235, dtype=np.uint8)
    pixels[:, :4] = 0
    y1, y2, x1, x2 = 180, 580, 80, 680
    yy, xx = np.mgrid[y1:y2, x1:x2]
    texture = ((xx * 7 + yy * 11) % 150).astype(np.uint8)
    pixels[y1:y2, x1:x2] = np.stack((texture, texture, texture), axis=2)
    for x in range(4, x1, 4):
        pixels[y1, x] = 0

    mask = _photographic_region_mask(pixels)

    assert mask[(y1 + y2) // 2, (x1 + x2) // 2] == 255


def test_source_cleanup_keeps_large_shaded_form_panel_exact() -> None:
    pixels = np.full((1000, 800, 3), (235, 225, 205), dtype=np.uint8)
    y1, y2, x1, x2 = 180, 780, 560, 740
    yy, xx = np.mgrid[y1:y2, x1:x2]
    shade = (190 + ((xx + yy) % 7)).astype(np.uint8)
    pixels[y1:y2, x1:x2] = np.stack((shade, shade, shade), axis=2)
    source = Image.fromarray(pixels, "RGB")

    cleaned = np.asarray(source_preserving_cleanup(source))

    assert np.array_equal(cleaned[y1:y2, x1:x2], pixels[y1:y2, x1:x2])


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
