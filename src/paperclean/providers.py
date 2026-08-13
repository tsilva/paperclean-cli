"""Provider-neutral model client contracts used by the cleaning pipeline."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol, Self

from PIL import Image

from paperclean.models import PageGeometry, ReviewVerdict
from paperclean.preflight import CostProjection


class UsageTracker(Protocol):
    total: Decimal
    ambiguous_timeouts: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ModelClient(Protocol):
    costs: UsageTracker
    backend_version: str | None

    def __enter__(self) -> Self: ...

    def __exit__(self, *_: object) -> None: ...

    def preflight(self) -> None: ...

    def cost_projection(
        self,
        *,
        document_total: int,
        page_total: int,
        max_attempts: int,
    ) -> CostProjection: ...

    def generate(self, source: Image.Image, prompt: str, *, max_edge: int) -> Image.Image: ...

    def locate_page(self, source: Image.Image) -> PageGeometry | None: ...

    def review(
        self,
        source: Image.Image,
        candidate: Image.Image,
        *,
        view_name: str,
    ) -> ReviewVerdict: ...
