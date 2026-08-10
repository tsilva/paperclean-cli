from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from paperclean.validation import OcrToken, _foreground_issues, _registered_token_issues


def _token(text: str, x: float, y: float) -> OcrToken:
    return OcrToken(text=text, confidence=95, box=(x, y, x + 0.02, y + 0.02))


def test_token_matching_uses_nearest_duplicate_instead_of_ocr_order() -> None:
    source = [_token("same", 0.1, 0.1), _token("same", 0.8, 0.8)]
    candidate = [_token("same", 0.8, 0.8), _token("same", 0.1, 0.1)]

    assert _registered_token_issues(source, candidate, np.eye(3)) == []


def test_token_matching_tolerates_small_ocr_segmentation_noise() -> None:
    source = [_token(str(index), index / 20, 0.5) for index in range(10)]
    candidate = source[:-1]

    assert _registered_token_issues(source, candidate, np.eye(3)) == []


def test_token_matching_rejects_material_text_loss() -> None:
    source = [_token(str(index), index / 20, 0.5) for index in range(10)]
    candidate = source[:-2]

    assert _registered_token_issues(source, candidate, np.eye(3)) == [
        "missing_or_changed_high_confidence_text"
    ]


def test_token_matching_distinguishes_moved_text() -> None:
    source = [_token(str(index), index / 20, 0.1) for index in range(10)]
    candidate = [_token(str(index), index / 20, 0.5) for index in range(10)]

    assert _registered_token_issues(source, candidate, np.eye(3)) == ["high_confidence_text_moved"]


def test_foreground_check_tolerates_restored_stroke_shape() -> None:
    source = Image.new("RGB", (400, 500), "white")
    candidate = source.copy()
    ImageDraw.Draw(source).rectangle((80, 200, 320, 208), fill="black")
    ImageDraw.Draw(candidate).rectangle((80, 204, 320, 214), fill="black")

    assert _foreground_issues(source, candidate, np.eye(3)) == []


def test_foreground_check_rejects_material_content_loss() -> None:
    source = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(source)
    for y in range(100, 401, 30):
        draw.rectangle((60, y, 340, y + 8), fill="black")
    candidate = Image.new("RGB", source.size, "white")

    assert "large_foreground_loss" in _foreground_issues(source, candidate, np.eye(3))
