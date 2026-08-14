"""Small, serializable domain records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PageStatus = Literal[
    "model_generated_clean",
    "model_assisted_clean",
    "source_preserving_clean",
    "original_fallback",
]
AttemptStrategy = Literal[
    "model_generation",
    "model_assisted_source_cleanup",
    "source_preserving_cleanup",
]
DiscrepancyCategory = Literal[
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
]


@dataclass(slots=True)
class UsageRecord:
    cost_usd: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class Discrepancy:
    category: str
    severity: Literal["low", "medium", "high", "critical"]
    region: tuple[float, float, float, float]


@dataclass(slots=True)
class ReviewVerdict:
    content_match: bool
    scanner_quality: bool
    discrepancies: list[Discrepancy] = field(default_factory=list)
    usage: UsageRecord = field(default_factory=UsageRecord)

    @property
    def accepted(self) -> bool:
        return self.content_match and self.scanner_quality and not self.discrepancies


@dataclass(frozen=True, slots=True)
class PageGeometry:
    corners: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    content_corners: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    occlusions: tuple[tuple[tuple[float, float], ...], ...]
    confidence: float
    page_polygon: tuple[tuple[float, float], ...] = ()
    edge_content: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass(slots=True)
class AttemptRecord:
    number: int
    strategy: AttemptStrategy = "model_generation"
    local_issues: list[str] = field(default_factory=list)
    verification_categories: list[str] = field(default_factory=list)
    generated_width: int | None = None
    generated_height: int | None = None
    effective_dpi: float | None = None
    accepted: bool = False
    error_type: str | None = None
    verification_discrepancies: list[Discrepancy] = field(default_factory=list)
    localized_quality_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    committed_quality_regions: list[tuple[float, float, float, float]] = field(default_factory=list)
    rejected_quality_regions: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class PageRecord:
    page: int
    status: PageStatus
    source_render_sha256: str
    final_render_sha256: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    fallback_reason: str | None = None


@dataclass(slots=True)
class DocumentReport:
    schema_version: int
    run_id: str
    source: str
    output: str
    source_sha256: str
    output_sha256: str | None
    backend: Literal["openrouter", "agentbridge"]
    billing_mode: Literal["openrouter_usd", "codex_subscription"]
    image_model: str
    verification_model: str
    verification_strategy: str
    started_at: str
    finished_at: str | None = None
    pages: list[PageRecord] = field(default_factory=list)
    backend_version: str | None = None
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ambiguous_timeout_charges: int = 0
    removed_pdf_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
