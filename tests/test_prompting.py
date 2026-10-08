from __future__ import annotations

import pytest

from paperclean.prompting import (
    CANDIDATE_QUALITY_PROMPT,
    FEEDBACK_TEMPLATE,
    GENERATION_PROMPT,
    ORIENTATION_PROMPT,
    PAGE_LOCATION_PROMPT,
    PHOTO_RECTIFICATION_PROMPT,
    PUNCH_HOLE_REPAIR_PROMPT,
    REGIONAL_REPAIR_PROMPT,
    REVIEW_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    load_prompt,
)


def test_all_primary_prompts_load_from_packaged_markdown() -> None:
    assert load_prompt("candidate-quality.md") == CANDIDATE_QUALITY_PROMPT
    assert load_prompt("generation.md") == GENERATION_PROMPT
    assert load_prompt("orientation.md") == ORIENTATION_PROMPT
    assert load_prompt("page-location.md") == PAGE_LOCATION_PROMPT
    assert load_prompt("photo-rectification.md") == PHOTO_RECTIFICATION_PROMPT
    assert load_prompt("punch-hole-repair.md") == PUNCH_HOLE_REPAIR_PROMPT
    assert load_prompt("regional-repair.md") == REGIONAL_REPAIR_PROMPT
    assert load_prompt("review.md") == REVIEW_PROMPT
    assert load_prompt("review-system.md") == REVIEW_SYSTEM_PROMPT
    assert load_prompt("feedback.md") == FEEDBACK_TEMPLATE


def test_review_prompt_requires_unambiguous_punch_hole_reconstruction() -> None:
    assert "punch hole overlaps authored ink" in REVIEW_PROMPT


def test_orientation_prompt_defines_one_unambiguous_whole_page_rotation() -> None:
    assert "counter-clockwise" in ORIENTATION_PROMPT
    assert "0, 90, 180, or 270" in ORIENTATION_PROMPT
    assert "Do not rotate an individual" in ORIENTATION_PROMPT


def test_review_prompt_rejects_non_upright_reading_orientation() -> None:
    assert "90,\n180, or 270 degree reading rotation" in REVIEW_PROMPT
    assert "merely plausible completion is not enough" in REVIEW_PROMPT
    assert "preserves the uncertain source evidence" in REVIEW_PROMPT


def test_review_prompt_explains_artificial_regional_crop_boundaries() -> None:
    assert "exact same intentional" in REVIEW_PROMPT and "crop from a larger page" in REVIEW_PROMPT
    assert "artificial verification-tile edges" in REVIEW_PROMPT
    assert "Never report" in REVIEW_PROMPT and "cropped_content" in REVIEW_PROMPT


def test_review_prompt_distinguishes_print_characteristics_from_scan_defects() -> None:
    assert "halftone logo texture" in REVIEW_PROMPT
    assert "not scan\ndefects" in REVIEW_PROMPT
    assert "tight actionable boxes" in REVIEW_PROMPT
    assert "full-page umbrella box" in REVIEW_PROMPT


def test_candidate_quality_prompt_is_source_independent_and_evidence_safe() -> None:
    assert "Inspect only the cleaned CANDIDATE" in CANDIDATE_QUALITY_PROMPT
    assert "not a comparison with\nthe original" in CANDIDATE_QUALITY_PROMPT
    assert "Preserved authored evidence is not a capture defect" in CANDIDATE_QUALITY_PROMPT
    assert "only scanner_quality discrepancies" in CANDIDATE_QUALITY_PROMPT


@pytest.mark.parametrize(
    "category",
    [
        "changed_text",
        "missing_text",
        "invented_text",
        "changed_handwriting",
        "changed_signature",
        "changed_stamp",
        "changed_redaction",
        "changed_table",
        "changed_diagram",
        "changed_layout",
        "cropped_content",
        "scanner_quality",
        "unresolved_content",
        "other_content",
    ],
)
def test_all_feedback_prompts_load_from_packaged_markdown(category: str) -> None:
    assert load_prompt(f"feedback/{category}.md").strip()


@pytest.mark.parametrize("name", ["generation.txt", "../generation.md", "/generation.md"])
def test_prompt_loader_rejects_non_resource_paths(name: str) -> None:
    with pytest.raises(ValueError):
        load_prompt(name)
