from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from paperclean.models import Discrepancy, PageGeometry
from paperclean.prompting import PUNCH_HOLE_REPAIR_PROMPT
from paperclean.restoration import (
    REGIONAL_REPAIR_PROMPT,
    _photographic_region_mask,
    _punch_hole_candidates,
    _remove_low_information_edge_artifacts,
    align_candidate_to_source,
    authored_punch_hole_regions,
    best_repair_region,
    clear_page_border,
    detect_page_plane,
    dewarp_curved_page,
    has_preserved_photographic_regions,
    photographed_page_cleanup,
    rectify_page_geometry,
    rectify_photographed_page,
    rectify_source_to_reference,
    registered_review_pairs,
    remove_photo_capture_artifacts,
    repair_region,
    replace_with_source_evidence_regions,
    rescue_colored_marks,
    restore_source_evidence_regions,
    restore_source_regions,
    source_preserving_cleanup,
)


def test_dewarp_curved_page_straightens_content_following_bowed_surface() -> None:
    height, width = 500, 400
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    line_rows: list[int] = []
    for x in range(30, 371):
        normalized_x = (x - 200) / 170
        top = round(38 + 42 * normalized_x**2)
        bottom = round(465 - 8 * normalized_x**2)
        mask[top : bottom + 1, x] = 255
        pixels[top : bottom + 1, x] = (225, 228, 235)
        line_y = round(top + 0.32 * (bottom - top))
        line_rows.append(line_y)
        pixels[line_y - 2 : line_y + 3, x] = (25, 45, 170)

    result = np.asarray(dewarp_curved_page(Image.fromarray(pixels, "RGB"), mask))
    blue = (result[:, :, 2] > result[:, :, 0] + 70) & (result[:, :, 2] > 100)
    observed_rows = []
    for x in range(45, 356):
        rows = np.flatnonzero(blue[:, x])
        assert len(rows) > 0
        observed_rows.append(float(np.median(rows)))

    assert np.ptp(observed_rows) <= 3
    assert np.ptp(line_rows) > 25


def test_dewarp_curved_page_leaves_flat_page_pixel_identical() -> None:
    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 25, 370, 475), fill=(230, 230, 235))
    draw.line((60, 180, 340, 180), fill=(25, 45, 170), width=5)
    mask = np.zeros((500, 400), dtype=np.uint8)
    mask[25:476, 30:371] = 255

    result = dewarp_curved_page(image, mask)

    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_detect_and_rectify_photographed_page_crops_camera_surroundings() -> None:
    pixels = np.full((800, 600, 3), (35, 55, 70), dtype=np.uint8)
    corners = np.array([[100, 80], [520, 125], [555, 735], [65, 700]], dtype=np.int32)
    cv2.fillConvexPoly(pixels, corners, (240, 238, 232))
    cv2.line(pixels, (150, 250), (475, 280), (20, 20, 20), 12)
    source = Image.fromarray(pixels, "RGB")

    plane = detect_page_plane(source)
    rectified = rectify_photographed_page(source)

    assert plane is not None
    assert plane.confidence >= 0.72
    assert rectified is not None
    assert rectified.height > rectified.width
    border = np.asarray(rectified)[0]
    assert float(border.mean()) > 180


def test_rectify_source_to_reference_uses_reference_geometry_not_pixels(
    monkeypatch,
) -> None:
    source = Image.new("RGB", (200, 300), "white")
    ImageDraw.Draw(source).rectangle((70, 100, 130, 112), fill="black")
    reference = Image.new("RGB", source.size, "magenta")
    source_to_reference = np.array(
        [[0.8, 0.0, 0.1], [0.0, 0.8, 0.1], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: np.linalg.inv(source_to_reference),
    )

    rectified = rectify_source_to_reference(source, reference)

    assert rectified is not None
    pixels = np.asarray(rectified)
    assert not np.any(np.all(pixels == (255, 0, 255), axis=2))
    assert np.any(np.all(pixels < 60, axis=2))


def test_align_candidate_to_source_restores_source_lattice(monkeypatch) -> None:
    source = Image.new("RGB", (200, 300), "white")
    ImageDraw.Draw(source).rectangle((70, 100, 130, 112), fill="black")
    candidate = Image.new("RGB", source.size, "white")
    ImageDraw.Draw(candidate).rectangle((90, 120, 150, 132), fill="black")
    candidate_to_source = np.array(
        [[1.0, 0.0, -0.1], [0.0, 1.0, -0.0666667], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: candidate_to_source,
    )

    aligned = align_candidate_to_source(source, candidate)

    assert aligned is not None
    pixels = np.asarray(aligned)
    assert np.all(pixels[103:110, 75:125] < 20)


def test_restore_source_regions_can_skip_registration_for_aligned_pages(monkeypatch) -> None:
    source = Image.new("RGB", (200, 300), "white")
    ImageDraw.Draw(source).text((100, 220), "ID42", fill="black")
    candidate = Image.new("RGB", source.size, "white")
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: (_ for _ in ()).throw(AssertionError("registration should be skipped")),
    )

    restored = restore_source_regions(
        source,
        candidate,
        [(0.4, 0.65, 0.9, 0.9)],
        already_aligned=True,
    )

    assert np.count_nonzero(np.asarray(restored)[200:280, 80:190] < 100) > 5


def test_restore_source_evidence_can_skip_registration_for_aligned_pages(monkeypatch) -> None:
    source = Image.new("RGB", (100, 100), "white")
    source_pixels = np.asarray(source).copy()
    source_pixels[45:55, 45:55] = (150, 205, 225)
    source = Image.fromarray(source_pixels, "RGB")
    candidate = Image.new("RGB", source.size, "white")
    monkeypatch.setattr(
        "paperclean.restoration._registration_matrix",
        lambda *_args: (_ for _ in ()).throw(AssertionError("registration should be skipped")),
    )

    restored = restore_source_evidence_regions(
        source,
        candidate,
        [(0.4, 0.4, 0.6, 0.6)],
        already_aligned=True,
    )

    restored_pixels = np.asarray(restored)
    assert restored_pixels[45:55, 45:55, 0].mean() < 230
    assert restored_pixels[45:55, 45:55, 2].mean() > restored_pixels[45:55, 45:55, 0].mean()


def test_replace_with_source_evidence_removes_generated_pixels_and_dirty_background() -> None:
    source = Image.new("RGB", (200, 300), (220, 215, 205))
    source_pixels = np.asarray(source).copy()
    source_pixels[120:126, 60:140] = (25, 45, 175)
    source = Image.fromarray(source_pixels, "RGB")
    candidate = Image.new("RGB", source.size, "white")
    candidate_pixels = np.asarray(candidate).copy()
    candidate_pixels[90:180, 40:160] = (180, 180, 180)
    candidate_pixels[140:146, 60:140] = (0, 0, 0)

    restored = replace_with_source_evidence_regions(
        source,
        Image.fromarray(candidate_pixels, "RGB"),
        [(0.2, 0.25, 0.8, 0.6)],
        already_aligned=True,
    )

    pixels = np.asarray(restored)
    assert np.all(pixels[100, 100] == 255)
    assert pixels[123, 100, 2] > pixels[123, 100, 0]
    assert np.all(pixels[143, 100] == 255)


def test_rectify_page_geometry_removes_surroundings_and_flattens_source_pixels() -> None:
    source = Image.new("RGB", (400, 500), (20, 60, 90))
    polygon = [(70, 50), (350, 80), (370, 460), (45, 440)]
    draw = ImageDraw.Draw(source)
    draw.polygon(polygon, fill=(235, 232, 225))
    draw.line((100, 180, 320, 200), fill="black", width=8)
    geometry = PageGeometry(
        corners=((0.175, 0.10), (0.875, 0.16), (0.925, 0.92), (0.1125, 0.88)),
        content_corners=((0.175, 0.10), (0.875, 0.16), (0.925, 0.92), (0.1125, 0.88)),
        occlusions=(),
        confidence=0.95,
    )

    rectified = rectify_page_geometry(source, geometry)

    assert rectified is not None
    pixels = np.asarray(photographed_page_cleanup(rectified))
    assert np.all(pixels[0, 0] == 255)
    background_fraction = float(np.mean(np.all(pixels == (20, 60, 90), axis=2)))
    assert background_fraction < 0.002
    assert np.any(np.all(pixels < 50, axis=2))


def test_rectify_page_geometry_masks_curled_edges_and_border_occlusions() -> None:
    source = Image.new("RGB", (400, 500), (30, 55, 70))
    draw = ImageDraw.Draw(source)
    visible_page = [(70, 90), (180, 60), (300, 82), (350, 110), (365, 450), (45, 440)]
    draw.polygon(visible_page, fill=(235, 232, 225))
    draw.line((95, 200, 320, 220), fill=(20, 40, 170), width=6)
    draw.line((120, 433, 315, 444), fill=(20, 40, 170), width=3)
    geometry = PageGeometry(
        corners=((0.175, 0.12), (0.875, 0.16), (0.9125, 0.90), (0.1125, 0.88)),
        content_corners=((0.20, 0.18), (0.84, 0.20), (0.86, 0.86), (0.14, 0.84)),
        occlusions=(),
        confidence=0.95,
        page_polygon=tuple((x / 400, y / 500) for x, y in visible_page),
        edge_content=(((0.25, 0.82), (0.82, 0.82), (0.82, 0.91), (0.25, 0.91)),),
    )

    rectified = rectify_page_geometry(source, geometry)

    assert rectified is not None
    pixels = np.asarray(photographed_page_cleanup(rectified))
    assert np.count_nonzero(np.all(pixels == (30, 55, 70), axis=2)) <= 1
    assert np.any(pixels[:, :, 2] > pixels[:, :, 0] + 80)
    assert np.count_nonzero(pixels[430:, :, 2] > pixels[430:, :, 0] + 80) > 20


def test_rectify_page_geometry_restores_ink_not_paper_inside_edge_polygon() -> None:
    source = Image.new("RGB", (400, 500), (235, 232, 225))
    draw = ImageDraw.Draw(source)
    draw.rectangle((0, 0, 399, 70), fill=(80, 100, 150))
    draw.line((90, 55, 300, 55), fill=(20, 40, 170), width=3)
    geometry = PageGeometry(
        corners=((0.0, 0.0), (0.9975, 0.0), (0.9975, 0.998), (0.0, 0.998)),
        content_corners=((0.05, 0.05), (0.95, 0.05), (0.95, 0.95), (0.05, 0.95)),
        occlusions=(),
        confidence=0.95,
        page_polygon=((0.0, 0.0), (0.9975, 0.0), (0.9975, 0.998), (0.0, 0.998)),
        edge_content=(((0.20, 0.08), (0.80, 0.08), (0.80, 0.14), (0.20, 0.14)),),
    )

    rectified = rectify_page_geometry(source, geometry)

    assert rectified is not None
    pixels = np.asarray(photographed_page_cleanup(rectified))
    assert np.all(pixels[25, 200] == 255)
    assert np.count_nonzero(pixels[45:80, :, 2] > pixels[45:80, :, 0] + 80) > 20


def test_remove_photo_capture_artifacts_preserves_thin_blue_handwriting() -> None:
    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 400, 55), fill=(35, 45, 40))
    draw.ellipse((120, 430, 260, 520), fill=(190, 95, 70))
    draw.line((80, 250, 330, 270), fill=(20, 45, 170), width=5)

    cleaned = remove_photo_capture_artifacts(image)
    pixels = np.asarray(cleaned)

    assert np.all(pixels[20, 200] == 255)
    assert np.all(pixels[470, 190] == 255)
    assert pixels[260, 205, 2] > pixels[260, 205, 0]


def test_photographed_page_cleanup_whitens_texture_and_preserves_colored_ink() -> None:
    y, x = np.indices((500, 400))
    texture = (12 * np.sin(x / 7) + 9 * np.cos(y / 11)).astype(np.int16)
    pixels = np.full((500, 400, 3), 218, dtype=np.int16)
    pixels += texture[:, :, None]
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    draw = ImageDraw.Draw(image)
    draw.line((60, 230, 340, 250), fill=(25, 45, 180), width=6)
    draw.rectangle((80, 100, 320, 112), fill=(30, 30, 30))

    cleaned = photographed_page_cleanup(image)
    result = np.asarray(cleaned)

    assert np.all(result[30, 30] == 255)
    assert result[240, 200, 2] > result[240, 200, 0]
    assert np.all(result[105, 200] < 80)


def test_photographed_page_cleanup_does_not_turn_paper_boundary_into_ink() -> None:
    image = Image.new("RGB", (500, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon(
        ((80, 100), (430, 115), (420, 630), (65, 610)),
        fill=(185, 195, 220),
    )
    draw.line((110, 270, 380, 290), fill=(25, 45, 175), width=6)

    result = np.asarray(photographed_page_cleanup(image))

    boundary_band = result[120:600, 55:115]
    assert float(np.mean(np.all(boundary_band == 255, axis=2))) > 0.97
    assert np.count_nonzero(result[250:320, :, 2] > result[250:320, :, 0] + 70) > 100


def test_photographed_page_cleanup_removes_compact_corner_warp_remnants() -> None:
    image = Image.new("RGB", (500, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon(((20, 22), (52, 28), (23, 48)), fill=(105, 125, 175))
    draw.line((420, 35, 490, 120), fill=(25, 45, 175), width=5)

    result = np.asarray(photographed_page_cleanup(image))

    assert np.all(result[15:60, 10:65] >= 252)
    assert np.count_nonzero(result[:140, 400:, 2] > result[:140, 400:, 0] + 70) > 100


def test_photographed_page_cleanup_removes_neutral_shadow_but_keeps_black_print() -> None:
    image = Image.new("RGB", (500, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 80, 450, 200), fill=(165, 165, 165))
    draw.rectangle((110, 330, 390, 345), fill=(25, 25, 25))

    result = np.asarray(photographed_page_cleanup(image))

    assert float(np.mean(np.all(result[100:180, 80:420] == 255, axis=2))) > 0.98
    assert np.all(result[334:342, 130:370] < 80)


def test_photographed_page_cleanup_removes_pale_rail_but_keeps_edge_handwriting() -> None:
    image = Image.new("RGB", (500, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((7, 100, 10, 610), fill=(175, 175, 180))
    draw.line((8, 250, 55, 270), fill=(25, 45, 175), width=5)

    result = np.asarray(photographed_page_cleanup(image))

    assert np.all(result[120:220, 5:13] == 255)
    assert np.count_nonzero(result[235:285, :70, 2] > result[235:285, :70, 0] + 70) > 80


def test_photographed_page_cleanup_keeps_faint_bottom_right_microprint() -> None:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((430, 735), "FOOTER", fill=(105, 125, 175), stroke_width=1)
    draw.rectangle((590, 200, 593, 680), fill=(185, 185, 190))

    result = np.asarray(photographed_page_cleanup(image))

    footer = result[720:780, 410:595]
    assert np.count_nonzero(footer[:, :, 2] > footer[:, :, 0] + 35) > 40
    assert np.all(result[250:600, 588:596] == 255)


def test_photographed_page_cleanup_keeps_sparse_microprint_inside_canvas_corner() -> None:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text((520, 755), "RIAS", fill=(55, 75, 155), stroke_width=1)
    draw.polygon(((565, 715), (598, 744), (598, 715)), fill=(105, 125, 175))

    result = np.asarray(photographed_page_cleanup(image))

    footer = result[745:790, 510:570]
    assert np.count_nonzero(footer[:, :, 2] > footer[:, :, 0] + 35) > 20
    assert np.all(result[705:750, 570:600] >= 252)


def test_photographed_page_cleanup_removes_warm_hand_leaking_from_edge() -> None:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 760, 340, 860), fill=(245, 184, 145))
    draw.line((180, 680, 430, 700), fill=(25, 45, 175), width=5)

    result = np.asarray(photographed_page_cleanup(image))

    assert np.all(result[770:, 100:360] >= 252)
    assert np.count_nonzero(result[660:720, :, 2] > result[660:720, :, 0] + 70) > 100


def test_photographed_page_cleanup_removes_desaturated_hand_inside_rectified_margin() -> None:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((30, 170, 125, 420), fill=(178, 151, 139))
    draw.line((45, 300, 210, 325), fill=(25, 45, 175), width=5)

    result = np.asarray(photographed_page_cleanup(image))

    warm = result[180:410, 25:135]
    assert float(np.mean(np.all(warm >= 252, axis=2))) > 0.92
    assert np.count_nonzero(result[280:345, 35:230, 2] > result[280:345, 35:230, 0] + 70) > 100


def test_photographed_page_cleanup_removes_warm_halo_without_erasing_dark_or_blue_ink() -> None:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 140, 180, 390), fill=(205, 185, 175))
    draw.line((95, 220, 260, 245), fill=(25, 45, 175), width=5)
    draw.text((100, 300), "ID", fill=(30, 30, 30))

    result = np.asarray(photographed_page_cleanup(image))

    assert float(np.mean(np.all(result[150:385, 75:190] >= 252, axis=2))) > 0.80
    assert np.count_nonzero(result[200:265, 85:280, 2] > result[200:265, 85:280, 0] + 70) > 100
    assert np.count_nonzero(np.all(result[285:340, 85:150] < 80, axis=2)) > 5


def test_photographed_page_cleanup_removes_fold_shadow_beside_edge_writing() -> None:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon(((545, 0), (600, 0), (600, 85)), fill=(165, 170, 185))
    draw.line((465, 55, 595, 105), fill=(25, 45, 175), width=5)

    result = np.asarray(photographed_page_cleanup(image))

    shadow = result[:80, 555:]
    assert float(np.mean(np.all(shadow >= 252, axis=2))) > 0.99
    assert np.count_nonzero(result[35:125, 440:, 2] > result[35:125, 440:, 0] + 70) > 100


def test_photographed_page_cleanup_removes_pale_border_rail() -> None:
    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 18, 499), fill=(190, 195, 205))
    draw.line((60, 250, 340, 260), fill=(20, 45, 170), width=6)

    result = np.asarray(photographed_page_cleanup(image))

    assert np.all(result[:, :20] == 255)
    assert result[255, 200, 2] > result[255, 200, 0]


def test_photographed_page_cleanup_preserves_faint_stamp_and_thin_footer() -> None:
    image = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.line((90, 470, 310, 470), fill=(20, 45, 170), width=3)
    draw.rectangle((160, 380, 240, 430), outline=(185, 205, 235), width=2)

    result = np.asarray(photographed_page_cleanup(image))

    assert result[470, 200, 2] > result[470, 200, 0]
    assert result[400, 160, 2] > result[400, 160, 0]


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
        abs(center_x - 45) <= 3 and abs(center_y - 300) <= 3 and touches_authored_ink
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


def test_source_cleanup_keeps_complete_gradient_shaded_form_column() -> None:
    pixels = np.full((1200, 900, 3), (235, 225, 205), dtype=np.uint8)
    y1, y2, x1, x2 = 260, 1080, 640, 850
    yy = np.arange(y1, y2, dtype=np.float32)[:, None]
    shade = np.broadcast_to(205 - (yy - y1) * 12 / (y2 - y1), (y2 - y1, x2 - x1))
    shade = shade.astype(np.uint8)
    pixels[y1:y2, x1:x2] = np.stack((shade, shade, shade), axis=2)
    cv2.rectangle(pixels, (x1, y1), (x2 - 1, y2 - 1), (150, 150, 150), thickness=2)
    cv2.putText(
        pixels,
        "10-20",
        (x1 + 35, y1 + 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (50, 50, 50),
        2,
        cv2.LINE_AA,
    )
    source = Image.fromarray(pixels, "RGB")

    cleaned = np.asarray(source_preserving_cleanup(source))

    assert np.array_equal(cleaned[y1:y2, x1:x2], pixels[y1:y2, x1:x2])


def test_source_cleanup_does_not_preserve_low_detail_edge_shadow() -> None:
    pixels = np.full((1000, 800, 3), (235, 225, 205), dtype=np.uint8)
    yy, xx = np.mgrid[820:1000, 120:760]
    shadow = np.clip(200 + (xx - 120) * 8 / 640 + (yy - 820) * 4 / 180, 0, 255).astype(
        np.uint8
    )
    pixels[820:1000, 120:760] = np.stack((shadow, shadow, shadow), axis=2)

    cleaned = np.asarray(source_preserving_cleanup(Image.fromarray(pixels, "RGB")))

    assert np.all(cleaned[850:980, 180:700] == 255)


def test_source_cleanup_removes_shallow_mottled_edge_wedge() -> None:
    pixels = np.full((1000, 800, 3), 255, dtype=np.uint8)
    wedge = np.asarray(((40, 930), (380, 970), (460, 999), (20, 999)), dtype=np.int32)
    cv2.fillConvexPoly(pixels, wedge, (205, 205, 205))
    noise = np.random.default_rng(7).integers(-9, 10, size=(70, 440, 1), dtype=np.int16)
    region = pixels[930:1000, 20:460].astype(np.int16)
    wedge_mask = np.zeros((1000, 800), dtype=np.uint8)
    cv2.fillConvexPoly(wedge_mask, wedge, 1)
    noisy = np.clip(region + noise, 0, 255).astype(np.uint8)
    region_mask = wedge_mask[930:1000, 20:460] > 0
    pixels[930:1000, 20:460][region_mask] = noisy[region_mask]

    cleaned = _remove_low_information_edge_artifacts(pixels)

    assert np.all(cleaned[950:995, 80:400] == 255)


def test_source_cleanup_preserves_dark_edge_content_and_textured_panel() -> None:
    pixels = np.full((1000, 800, 3), 255, dtype=np.uint8)
    cv2.putText(
        pixels,
        "FOOTER 123",
        (20, 985),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (15, 15, 15),
        3,
        cv2.LINE_AA,
    )
    yy, xx = np.mgrid[0:65, 560:800]
    texture = ((xx * 11 + yy * 17) % 170).astype(np.uint8)
    pixels[0:65, 560:800] = np.stack((texture, texture, texture), axis=2)

    cleaned = _remove_low_information_edge_artifacts(pixels)

    assert np.any(cleaned[950:995, 20:280] < 80)
    assert np.array_equal(cleaned[0:65, 560:800], pixels[0:65, 560:800])


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
