"""Load packaged model prompts from editable Markdown resources."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import PurePosixPath


@cache
def load_prompt(name: str) -> str:
    """Return a packaged Markdown prompt with one trailing newline."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
        raise ValueError("prompt names must be relative Markdown paths")
    resource = files("paperclean").joinpath("prompts", *path.parts)
    return resource.read_text(encoding="utf-8").strip() + "\n"


GENERATION_PROMPT = load_prompt("generation.md")
REGIONAL_REPAIR_PROMPT = load_prompt("regional-repair.md")
REVIEW_SYSTEM_PROMPT = load_prompt("review-system.md")
REVIEW_PROMPT = load_prompt("review.md")
FEEDBACK_TEMPLATE = load_prompt("feedback.md")
