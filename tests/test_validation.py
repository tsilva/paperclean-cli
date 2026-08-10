from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from paperclean.validation import (
    _foreground_issues,
    _has_meaningful_foreground,
    validate_candidate,
)


def test_blank_page_does_not_require_registration() -> None:
    blank = np.full((500, 400), 255, dtype=np.uint8)

    assert _has_meaningful_foreground(blank) is False


def test_authored_ink_requires_registration() -> None:
    image = Image.new("L", (400, 500), "white")
    ImageDraw.Draw(image).rectangle((80, 200, 320, 210), fill="black")

    assert _has_meaningful_foreground(np.asarray(image)) is True


def test_registration_failure_rejects_content_bearing_page(monkeypatch) -> None:
    source = Image.new("RGB", (400, 500), "white")
    ImageDraw.Draw(source).rectangle((80, 200, 320, 210), fill="black")
    monkeypatch.setattr("paperclean.validation._registration_matrix", lambda *_args: None)

    result = validate_candidate(
        source,
        source.copy(),
        min_effective_dpi=150,
        effective_dpi=300,
    )

    assert "page_registration_failed" in result.issues


def test_registration_failure_allows_genuinely_blank_page(monkeypatch) -> None:
    blank = Image.new("RGB", (400, 500), "white")
    monkeypatch.setattr("paperclean.validation._registration_matrix", lambda *_args: None)

    result = validate_candidate(
        blank,
        blank.copy(),
        min_effective_dpi=150,
        effective_dpi=300,
    )

    assert result.accepted is True


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
