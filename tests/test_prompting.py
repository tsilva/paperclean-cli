from __future__ import annotations

import pytest

from paperclean.prompting import (
    FEEDBACK_TEMPLATE,
    GENERATION_PROMPT,
    REGIONAL_REPAIR_PROMPT,
    REVIEW_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    load_prompt,
)


def test_all_primary_prompts_load_from_packaged_markdown() -> None:
    assert load_prompt("generation.md") == GENERATION_PROMPT
    assert load_prompt("regional-repair.md") == REGIONAL_REPAIR_PROMPT
    assert load_prompt("review.md") == REVIEW_PROMPT
    assert load_prompt("review-system.md") == REVIEW_SYSTEM_PROMPT
    assert load_prompt("feedback.md") == FEEDBACK_TEMPLATE


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
