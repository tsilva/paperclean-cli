"""End-to-end conservative document-cleaning orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from PIL import Image

from paperclean.config import Settings
from paperclean.discovery import OutputPaths
from paperclean.errors import (
    ContentPolicyError,
    CostLimitReached,
    GlobalProviderError,
    PayloadTooLargeError,
    ProviderError,
    ReviewerResponseError,
)
from paperclean.imaging import (
    final_pixel_image,
    finish_pristine_recreation,
    load_image,
    normalize_generated,
    pixel_sha256,
    review_boxes,
    review_view_pairs,
    source_dpi,
)
from paperclean.models import (
    AttemptRecord,
    Discrepancy,
    DocumentReport,
    PageRecord,
    ReviewVerdict,
)
from paperclean.pdfs import build_pdf, inspect_pdf, render_overlay_preview, render_pages
from paperclean.prompting import (
    FEEDBACK_TEMPLATE,
    GENERATION_PROMPT,
    PHOTO_RECTIFICATION_PROMPT,
    PUNCH_HOLE_REPAIR_PROMPT,
    load_prompt,
)
from paperclean.provenance import embed_image, manifest_wrapper, write_report
from paperclean.providers import ModelClient
from paperclean.restoration import (
    align_candidate_to_source,
    authored_punch_hole_regions,
    best_repair_region,
    clear_page_border,
    detect_page_plane,
    erase_contained_edge_artifacts,
    erase_localized_pale_artifacts,
    erase_residual_punch_hole_regions,
    has_preserved_photographic_regions,
    photographed_page_cleanup,
    rectify_page_geometry,
    rectify_source_to_reference,
    regions_are_preserved_visual_panels,
    registered_review_pairs,
    repair_region,
    replace_with_source_evidence_regions,
    rescue_colored_marks,
    residual_punch_hole_regions,
    restore_source_evidence_regions,
    restore_source_regions,
    rotate_reading_orientation,
    source_preserving_cleanup,
)
from paperclean.util import (
    private_workdir,
    private_write,
    publish_pair,
    sha256_file,
    staged_path,
)
from paperclean.validation import DeterministicResult, validate_candidate


@dataclass(slots=True)
class PageOutcome:
    output_image: Image.Image
    record: PageRecord
    cost_stopped: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _error_name(exc: BaseException) -> str:
    if isinstance(exc, ProviderError) and exc.error_type:
        return exc.error_type[:80]
    return type(exc).__name__


def _feedback(categories: list[str]) -> str:
    unique = list(dict.fromkeys(categories))[:5]
    lines = []
    for category in unique:
        try:
            lines.append(load_prompt(f"feedback/{category}.md").strip())
        except FileNotFoundError:
            continue
    requirements = "\n".join(f"- {line}" for line in lines)
    return "\n" + FEEDBACK_TEMPLATE.replace("{requirements}", requirements) if lines else ""


VERIFICATION_STRATEGY = "full-page-plus-four-registered-regions"


def _review_view(
    client: ModelClient,
    source: Image.Image,
    candidate: Image.Image,
    *,
    view_name: str,
) -> ReviewVerdict:
    for timeout_attempt in range(2):
        try:
            for schema_attempt in range(2):
                try:
                    return client.review(source, candidate, view_name=view_name)
                except ReviewerResponseError:
                    if schema_attempt:
                        raise
        except ProviderError as exc:
            if exc.error_type not in {"timeout", "timeout_error"} or timeout_attempt:
                raise
    raise ReviewerResponseError("reviewer did not produce a verdict")


def _quality_only_rejection(verdict: ReviewVerdict) -> bool:
    return (
        not verdict.accepted
        and verdict.content_match
        and all(item.category == "scanner_quality" for item in verdict.discrepancies)
    )


def _verification_accepts(
    client: ModelClient,
    source: Image.Image,
    candidate: Image.Image,
    *,
    tolerated_categories: frozenset[str] = frozenset(),
    confirm_rejections: bool = False,
    quality_consensus: bool = False,
    confirm_quality_rejections: bool = True,
    register_views: bool = True,
    collect_all_views: bool = False,
) -> tuple[bool, list[Discrepancy]]:
    discrepancies: list[Discrepancy] = []
    rejected = False
    view_pairs = (
        registered_review_pairs(source, candidate)
        if register_views
        else review_view_pairs(source, candidate)
    )
    for index, (source_view, candidate_view) in enumerate(view_pairs):
        view_name = "full page" if index == 0 else f"region {index} of 4"
        verdict = _review_view(
            client,
            source_view,
            candidate_view,
            view_name=view_name,
        )
        if quality_consensus and _quality_only_rejection(verdict):
            # Scan appearance is a subjective vision judgment and may vary across
            # otherwise identical calls. Require a two-of-three consensus before a
            # quality-only rejection can fail a page. Content mismatches do not use
            # this relaxation: they remain fail-closed and are only rechecked once
            # when the caller explicitly asks to confirm rejections.
            quality_verdicts = [
                verdict,
                _review_view(client, source_view, candidate_view, view_name=view_name),
                _review_view(client, source_view, candidate_view, view_name=view_name),
            ]
            accepted_quality = [item for item in quality_verdicts if item.accepted]
            if len(accepted_quality) >= 2:
                verdict = accepted_quality[0]
            else:
                rejected_quality = [
                    item for item in quality_verdicts if _quality_only_rejection(item)
                ]
                verdict = rejected_quality[0] if rejected_quality else quality_verdicts[-1]
        elif (confirm_quality_rejections and _quality_only_rejection(verdict)) or (
            confirm_rejections and not verdict.accepted and not _quality_only_rejection(verdict)
        ):
            verdict = _review_view(
                client,
                source_view,
                candidate_view,
                view_name=view_name,
            )
        if (
            _quality_only_rejection(verdict)
            and verdict.discrepancies
            and regions_are_preserved_visual_panels(
                source_view,
                candidate_view,
                [item.region for item in verdict.discrepancies],
            )
        ):
            # A radiograph, chart, or authored shaded form panel is expected to
            # retain its intrinsic tone and texture. Ignore only explicit regions
            # wholly inside the same independently detected panel in both views.
            verdict = ReviewVerdict(
                content_match=True,
                scanner_quality=True,
                usage=verdict.usage,
            )
        view_discrepancies = list(verdict.discrepancies)
        if not verdict.content_match and not view_discrepancies:
            view_discrepancies.append(
                Discrepancy("unresolved_content", "high", (0.0, 0.0, 1.0, 1.0))
            )
        if not verdict.scanner_quality and not any(
            item.category == "scanner_quality" for item in view_discrepancies
        ):
            view_discrepancies.append(Discrepancy("scanner_quality", "high", (0.0, 0.0, 1.0, 1.0)))
        view_box = (
            (0, 0, candidate.width, candidate.height)
            if index == 0
            else review_boxes(candidate.size)[index - 1]
        )
        view_left, view_top, view_right, view_bottom = view_box
        view_width = view_right - view_left
        view_height = view_bottom - view_top
        for item in view_discrepancies:
            left, top, right, bottom = item.region
            discrepancies.append(
                Discrepancy(
                    category=item.category,
                    severity=item.severity,
                    region=(
                        (view_left + left * view_width) / candidate.width,
                        (view_top + top * view_height) / candidate.height,
                        (view_left + right * view_width) / candidate.width,
                        (view_top + bottom * view_height) / candidate.height,
                    ),
                )
            )
        tolerated_transformation = bool(verdict.discrepancies) and all(
            item.category in tolerated_categories for item in verdict.discrepancies
        )
        if not verdict.accepted and not tolerated_transformation:
            rejected = True
            if not collect_all_views:
                return False, discrepancies
    return not rejected, discrepancies


def _incremental_content_accepts(
    client: ModelClient,
    source: Image.Image,
    candidate: Image.Image,
    region: tuple[float, float, float, float],
    *,
    tolerated_categories: frozenset[str] = frozenset(),
) -> bool:
    """Check one incremental repair without repeating full quality consensus.

    The initial verification has already localized scan-quality defects. During a
    repair transaction, only newly changed content can invalidate the proposal;
    scanner quality is adjudicated once after the bounded queue is exhausted.
    Review the full page plus the registered verification tile with the greatest
    overlap, confirming a content rejection once to avoid one noisy judgment.
    """

    def content_matches(verdict: ReviewVerdict) -> bool:
        return verdict.content_match or (
            bool(verdict.discrepancies)
            and all(item.category in tolerated_categories for item in verdict.discrepancies)
        )

    view_pairs = registered_review_pairs(source, candidate)
    views: list[tuple[Image.Image, Image.Image, str]] = [
        (view_pairs[0][0], view_pairs[0][1], "full page")
    ]
    boxes = review_boxes(candidate.size)
    left, top, right, bottom = region

    def overlap_area(box: tuple[int, int, int, int]) -> float:
        box_left, box_top, box_right, box_bottom = box
        normalized = (
            box_left / candidate.width,
            box_top / candidate.height,
            box_right / candidate.width,
            box_bottom / candidate.height,
        )
        return max(0.0, min(right, normalized[2]) - max(left, normalized[0])) * max(
            0.0,
            min(bottom, normalized[3]) - max(top, normalized[1]),
        )

    tile_index = max(range(len(boxes)), key=lambda index: overlap_area(boxes[index]))
    views.append(
        (
            view_pairs[tile_index + 1][0],
            view_pairs[tile_index + 1][1],
            f"region {tile_index + 1} of 4",
        )
    )
    for source_view, candidate_view, view_name in views:
        verdict = _review_view(client, source_view, candidate_view, view_name=view_name)
        if content_matches(verdict):
            continue
        confirmed = _review_view(client, source_view, candidate_view, view_name=view_name)
        if not content_matches(confirmed):
            return False
    return True


_SOURCE_PRESERVED_CATEGORIES = {
    "changed_diagram",
    "changed_handwriting",
    "changed_layout",
    "changed_signature",
    "changed_stamp",
    "changed_redaction",
    "changed_text",
    "changed_table",
    "cropped_content",
    "invented_text",
    "missing_text",
    "unresolved_content",
}

_SOURCE_CLEANUP_TOLERATED_CATEGORIES = frozenset(
    {
        "changed_layout",
        "other_content",
        "unresolved_content",
    }
)
_MODEL_ASSISTED_CLEANUP_TOLERATED_CATEGORIES = frozenset(
    {
        "changed_layout",
    }
)
_MAX_AUTHORED_HOLE_REPAIRS = 2
_MAX_AUTHORED_HOLE_REPAIR_ATTEMPTS = 3
_MAX_SOURCE_REGION_RECOVERY_PASSES = 4
_MAX_PHOTO_QUALITY_REPAIRS = 2
_MAX_SCAN_QUALITY_REPAIRS = 4
_SOURCE_EVIDENCE_RECOVERY_CATEGORIES = {
    "changed_text",
    "missing_text",
    "invented_text",
    "changed_handwriting",
    "changed_signature",
    "changed_stamp",
    "changed_redaction",
    "changed_table",
    "cropped_content",
}


def _preserve_from_source(discrepancy: Discrepancy) -> bool:
    return discrepancy.category in _SOURCE_PRESERVED_CATEGORIES


def _source_region(discrepancy: Discrepancy) -> tuple[float, float, float, float]:
    left, top, right, bottom = discrepancy.region
    if bottom > 0.94:
        return (0.0, min(top, 0.94), 1.0, 1.0)
    return (left, top, right, bottom)


def _localized_quality_repair_region(
    discrepancies: list[Discrepancy],
) -> tuple[float, float, float, float] | None:
    """Return one bounded scan-quality region, never a broad page recreation.

    Quality boxes often describe a long, shallow fold or a narrow scanner rail.
    ``best_repair_region`` deliberately adds proportional context for missing
    content, but that padding can turn a genuinely thin defect into an apparently
    broad region. Use small anisotropic context here so thin strips remain eligible
    without allowing a model to redraw a large two-dimensional page area.
    """
    quality = [item for item in discrepancies if item.category == "scanner_quality"]
    if not quality:
        return None
    severity_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    selected = max(quality, key=lambda item: severity_rank.get(item.severity, 0))
    left, top, right, bottom = selected.region
    raw_width = right - left
    raw_height = bottom - top
    if raw_width <= 0 or raw_height <= 0:
        return None
    pad_x = max(0.01, min(0.03, raw_width * 0.08))
    pad_y = max(0.01, min(0.04, raw_height * 0.50))
    region = (
        max(0.0, left - pad_x),
        max(0.0, top - pad_y),
        min(1.0, right + pad_x),
        min(1.0, bottom + pad_y),
    )
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    if (
        width * height > 0.15
        or (width > 0.75 and height > 0.10)
        or (height > 0.65 and width > 0.10)
        or (width > 0.55 and height > 0.15)
        or (height > 0.55 and width > 0.15)
    ):
        return None
    return region


def _expanded_quality_repair_context(
    region: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Give a thin quality defect enough perpendicular context to reconstruct.

    A fold or scanner rail may be only a few pixels tall or wide. Asking an image
    model to recreate that sliver deprives it of complete glyphs and baselines, so
    even an otherwise faithful repair can rewrite or clip text. Expand only the
    contextual crop to at least twelve percent of the page in the thin dimension;
    :func:`repair_region` still splices back only the original bounded region.
    """
    left, top, right, bottom = region
    width = right - left
    height = bottom - top
    horizontal = height <= 0.12 and width >= 0.20
    vertical = width <= 0.12 and height >= 0.20
    if horizontal and height < 0.12:
        center = (top + bottom) / 2
        top = max(0.0, center - 0.06)
        bottom = min(1.0, top + 0.12)
        top = max(0.0, bottom - 0.12)
    elif vertical and width < 0.12:
        center = (left + right) / 2
        left = max(0.0, center - 0.06)
        right = min(1.0, left + 0.12)
        left = max(0.0, right - 0.12)
    return (left, top, right, bottom)


def _regions_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def _expanded_hole_paste_region(
    context: tuple[float, float, float, float],
    defect: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Expand a physical-hole footprint enough to splice whole nearby glyphs.

    A circle-sized paste can clip a correctly reconstructed word at its boundary,
    especially for holes along the binding edge. Keep the splice inside the
    detector's contextual region while including a bounded horizontal word span and
    vertical line span. Local and whole-page semantic review still arbitrate it.
    """
    context_left, context_top, context_right, context_bottom = context
    left, top, right, bottom = defect
    target_width = min(
        context_right - context_left,
        max(0.12, (right - left) * 3.0),
    )
    target_height = min(
        context_bottom - context_top,
        max(0.05, (bottom - top) * 2.0),
    )
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    paste_left = max(context_left, center_x - target_width / 2)
    paste_right = min(context_right, paste_left + target_width)
    paste_left = max(context_left, paste_right - target_width)
    paste_top = max(context_top, center_y - target_height / 2)
    paste_bottom = min(context_bottom, paste_top + target_height)
    paste_top = max(context_top, paste_bottom - target_height)
    return (paste_left, paste_top, paste_right, paste_bottom)


def _expanded_hole_repair_context(
    context: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Provide whole neighboring lines and words to punch-hole generation.

    Punch detection deliberately returns a tight line-level crop. That is ideal
    for verification and the eventual splice, but a very shallow generation crop
    can contain only partial glyphs and make faithful baseline reconstruction
    unstable. Expand the model's reference context while retaining the separately
    bounded paste region computed by :func:`_expanded_hole_paste_region`.
    """
    left, top, right, bottom = context
    target_width = max(right - left, 0.36)
    target_height = max(bottom - top, 0.12)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    expanded_left = max(0.0, center_x - target_width / 2)
    expanded_right = min(1.0, expanded_left + target_width)
    expanded_left = max(0.0, expanded_right - target_width)
    expanded_top = max(0.0, center_y - target_height / 2)
    expanded_bottom = min(1.0, expanded_top + target_height)
    expanded_top = max(0.0, expanded_bottom - target_height)
    return (expanded_left, expanded_top, expanded_right, expanded_bottom)


def _match_repair_regions(
    contexts: list[tuple[float, float, float, float]],
    defects: list[tuple[float, float, float, float]],
) -> list[
    tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]
]:
    """Match unordered defect footprints to their nearest contextual regions."""
    remaining = list(contexts)
    matched: list[
        tuple[
            tuple[float, float, float, float],
            tuple[float, float, float, float],
        ]
    ] = []
    for defect in defects:
        if not remaining:
            break
        defect_center = ((defect[0] + defect[2]) / 2, (defect[1] + defect[3]) / 2)
        context = min(
            remaining,
            key=lambda item: (
                ((item[0] + item[2]) / 2 - defect_center[0]) ** 2
                + ((item[1] + item[3]) / 2 - defect_center[1]) ** 2
            ),
        )
        remaining.remove(context)
        matched.append((context, defect))
    return matched


def _photo_local_gate_accepts(issues: list[str]) -> bool:
    """Allow semantic review to arbitrate apparent foreground loss in photos.

    Adaptive local thresholding cannot reliably distinguish printed strokes from
    hard fold and cast-shadow edges on a crumpled photographed sheet.  It remains a
    useful catastrophic-loss warning, but it must not prevent the five registered
    vision comparisons from identifying the actual missing text, codes, or marks.
    Geometry, registration, resolution, and candidate-only foreground failures stay
    hard blockers.
    """
    return all(issue == "large_foreground_loss" for issue in issues)


def _validate_clean_candidate(
    source: Image.Image,
    candidate: Image.Image,
    *,
    min_effective_dpi: int,
    effective_dpi: float,
) -> DeterministicResult:
    """Apply fidelity checks plus an explicit gate for surviving punch holes."""
    deterministic = validate_candidate(
        source,
        candidate,
        min_effective_dpi=min_effective_dpi,
        effective_dpi=effective_dpi,
    )
    issues = list(deterministic.issues)
    if residual_punch_hole_regions(source, candidate):
        issues.append("residual_punch_hole")
    issues = list(dict.fromkeys(issues))
    return DeterministicResult(accepted=not issues, issues=issues)


def _repair_authored_hole_regions(
    source: Image.Image,
    recreation: Image.Image,
    candidate: Image.Image,
    hole_regions: list[tuple[float, float, float, float]],
    *,
    settings: Settings,
    client: ModelClient,
    finalize_candidate: Callable[[Image.Image], Image.Image],
    source_page_dpi: float,
    prefer_deterministic_erase: bool = False,
    repair_erased_holes: bool = False,
) -> tuple[Image.Image, Image.Image, DeterministicResult]:
    """Repair source-authored punch regions before any full-page regeneration."""
    assisted_recreation = recreation
    residual_regions = residual_punch_hole_regions(source, candidate)
    if prefer_deterministic_erase and residual_regions:
        erased_recreation = erase_residual_punch_hole_regions(
            assisted_recreation,
            residual_regions,
        )
        erased_candidate = finalize_candidate(erased_recreation)
        erased_residual_regions = residual_punch_hole_regions(source, erased_candidate)
        if len(erased_residual_regions) < len(residual_regions):
            assisted_recreation = erased_recreation
            candidate = erased_candidate
            residual_regions = erased_residual_regions
    if repair_erased_holes:
        # A source-preserving cleanup can erase a circle completely while also
        # erasing authored strokes beneath it. Candidate-only residual detection
        # cannot see that case, so retain every physical footprint detected in the
        # source and reconstruct it under the same local semantic gate.
        source_regions = residual_punch_hole_regions(source, source)
        residual_regions = list(dict.fromkeys([*residual_regions, *source_regions]))
    for hole_region, residual_region in _match_repair_regions(
        hole_regions,
        residual_regions,
    ):
        paste_region = _expanded_hole_paste_region(hole_region, residual_region)
        repair_context = _expanded_hole_repair_context(hole_region)
        for _repair_attempt in range(_MAX_AUTHORED_HOLE_REPAIR_ATTEMPTS):
            proposed_recreation = repair_region(
                source,
                assisted_recreation,
                repair_context,
                client=client,
                max_edge=settings.max_reference_edge,
                prompt=PUNCH_HOLE_REPAIR_PROMPT,
                paste_region=paste_region,
            )
            proposed_candidate = finalize_candidate(proposed_recreation)
            remaining_holes = residual_punch_hole_regions(source, proposed_candidate)
            if not any(
                _regions_overlap(residual_region, remaining) for remaining in remaining_holes
            ):
                left, top, right, bottom = hole_region
                crop_box = (
                    round(left * source.width),
                    round(top * source.height),
                    round(right * source.width),
                    round(bottom * source.height),
                )
                local_verdict = _review_view(
                    client,
                    source.crop(crop_box),
                    proposed_candidate.crop(crop_box),
                    view_name="region 1 of 1",
                )
                if not local_verdict.content_match:
                    # A single vision judgment can be noisy around deliberately
                    # reconstructed, previously occluded strokes. Confirm a local
                    # rejection once before discarding the generated crop. This
                    # does not publish the candidate: deterministic checks and the
                    # full registered page comparison remain mandatory afterward.
                    local_verdict = _review_view(
                        client,
                        source.crop(crop_box),
                        proposed_candidate.crop(crop_box),
                        view_name="region 1 of 1",
                    )
                if local_verdict.content_match:
                    assisted_recreation = proposed_recreation
                    break
    assisted_candidate = finalize_candidate(assisted_recreation)
    assisted_deterministic = _validate_clean_candidate(
        source,
        assisted_candidate,
        min_effective_dpi=settings.min_effective_dpi,
        effective_dpi=source_page_dpi,
    )
    return assisted_recreation, assisted_candidate, assisted_deterministic


def _try_source_first_cleanup(
    source: Image.Image,
    *,
    page_number: int,
    source_hash: str,
    source_page_dpi: float,
    settings: Settings,
    client: ModelClient,
    finalize_candidate: Callable[[Image.Image], Image.Image],
    attempts: list[AttemptRecord],
) -> PageOutcome | None:
    """Verify deterministic scan cleanup before requesting generative recreation.

    AgentBridge image generation is substantially slower and less reliable than its
    structured vision review.  Ordinary scans normally need photometric cleanup,
    deskewing, border removal, and punch-hole removal—not regenerated glyphs.  Try
    that source-derived candidate first and reserve generation for a verified
    failure. A hole touching authored ink is repaired through a bounded contextual
    crop here, before any full-page generation is considered.
    """
    record = AttemptRecord(
        number=len(attempts) + 1,
        strategy="source_preserving_cleanup",
        effective_dpi=round(source_page_dpi, 2),
    )
    attempts.append(record)
    recreation = source_preserving_cleanup(source)
    candidate = finalize_candidate(recreation)
    deterministic = _validate_clean_candidate(
        source,
        candidate,
        min_effective_dpi=settings.min_effective_dpi,
        effective_dpi=source_page_dpi,
    )
    record.local_issues = deterministic.issues
    hole_regions = authored_punch_hole_regions(source)[:_MAX_AUTHORED_HOLE_REPAIRS]
    if hole_regions and set(deterministic.issues) == {"residual_punch_hole"}:
        record.strategy = "model_assisted_source_cleanup"
        try:
            recreation, candidate, deterministic = _repair_authored_hole_regions(
                source,
                recreation,
                candidate,
                hole_regions,
                settings=settings,
                client=client,
                finalize_candidate=finalize_candidate,
                source_page_dpi=source_page_dpi,
                repair_erased_holes=True,
            )
        except (ContentPolicyError, ReviewerResponseError, ProviderError) as exc:
            record.error_type = _error_name(exc)
            raise
        record.local_issues = deterministic.issues
    if not deterministic.accepted:
        return None
    tolerated_categories = (
        _MODEL_ASSISTED_CLEANUP_TOLERATED_CATEGORIES
        if record.strategy == "model_assisted_source_cleanup"
        else _SOURCE_CLEANUP_TOLERATED_CATEGORIES
    )
    if has_preserved_photographic_regions(source):
        tolerated_categories = tolerated_categories | {"changed_diagram"}
    accepted, discrepancies = _verification_accepts(
        client,
        source,
        candidate,
        tolerated_categories=tolerated_categories,
        confirm_rejections=True,
        quality_consensus=False,
        confirm_quality_rejections=False,
        collect_all_views=True,
    )
    evidence_regions = [
        _source_region(item)
        for item in discrepancies
        if item.category in _SOURCE_EVIDENCE_RECOVERY_CATEGORIES
    ]
    if not accepted and evidence_regions:
        evidence_recreation = restore_source_evidence_regions(
            source,
            recreation,
            evidence_regions,
        )
        evidence_candidate = finalize_candidate(evidence_recreation)
        evidence_deterministic = _validate_clean_candidate(
            source,
            evidence_candidate,
            min_effective_dpi=settings.min_effective_dpi,
            effective_dpi=source_page_dpi,
        )
        record.local_issues = evidence_deterministic.issues
        if evidence_deterministic.accepted:
            accepted, discrepancies = _verification_accepts(
                client,
                source,
                evidence_candidate,
                tolerated_categories=tolerated_categories,
                confirm_rejections=True,
                quality_consensus=True,
                collect_all_views=True,
            )
            if accepted:
                recreation = evidence_recreation
                candidate = evidence_candidate
    if (
        not accepted
        and discrepancies
        and all(item.category == "scanner_quality" for item in discrepancies)
    ):
        quality_regions = [
            region
            for item in discrepancies
            if (region := _localized_quality_repair_region([item])) is not None
        ]
        quality_regions = list(dict.fromkeys(quality_regions))[:_MAX_SCAN_QUALITY_REPAIRS]
        record.localized_quality_regions = quality_regions
        committed_repair = False
        proposed_tolerated_categories = _MODEL_ASSISTED_CLEANUP_TOLERATED_CATEGORIES
        if has_preserved_photographic_regions(source):
            proposed_tolerated_categories = proposed_tolerated_categories | {"changed_diagram"}
        for quality_region in quality_regions:
            # Each region is an independent transaction against the current
            # content-safe candidate. Do not rediscover quality boxes after every
            # commit; the original all-view verdict is the bounded repair queue.
            deterministic_recreation = erase_contained_edge_artifacts(
                recreation,
                quality_region,
            )
            deterministic_recreation = erase_localized_pale_artifacts(
                deterministic_recreation,
                quality_region,
            )
            if pixel_sha256(deterministic_recreation) != pixel_sha256(recreation):
                proposed_recreation = deterministic_recreation
            else:
                repair_context = _expanded_quality_repair_context(quality_region)
                proposed_recreation = repair_region(
                    recreation,
                    recreation,
                    repair_context,
                    client=client,
                    max_edge=settings.max_reference_edge,
                    paste_region=quality_region,
                )
            if pixel_sha256(proposed_recreation) == pixel_sha256(recreation):
                record.rejected_quality_regions.append(quality_region)
                continue
            proposed_candidate = finalize_candidate(proposed_recreation)
            proposed_deterministic = _validate_clean_candidate(
                source,
                proposed_candidate,
                min_effective_dpi=settings.min_effective_dpi,
                effective_dpi=source_page_dpi,
            )
            record.local_issues = proposed_deterministic.issues
            if not proposed_deterministic.accepted:
                record.rejected_quality_regions.append(quality_region)
                continue
            if not _incremental_content_accepts(
                client,
                source,
                proposed_candidate,
                quality_region,
                tolerated_categories=proposed_tolerated_categories,
            ):
                record.rejected_quality_regions.append(quality_region)
                continue
            recreation = proposed_recreation
            candidate = proposed_candidate
            record.strategy = "model_assisted_source_cleanup"
            record.committed_quality_regions.append(quality_region)
            committed_repair = True
        if committed_repair or not quality_regions:
            accepted, discrepancies = _verification_accepts(
                client,
                source,
                candidate,
                tolerated_categories=proposed_tolerated_categories,
                confirm_rejections=True,
                quality_consensus=True,
                collect_all_views=True,
            )
    record.verification_categories = list(dict.fromkeys(item.category for item in discrepancies))
    record.verification_discrepancies = list(discrepancies)
    record.accepted = accepted
    if not accepted:
        return None
    return PageOutcome(
        output_image=recreation,
        record=PageRecord(
            page=page_number,
            status=(
                "model_assisted_clean"
                if record.strategy == "model_assisted_source_cleanup"
                else "source_preserving_clean"
            ),
            source_render_sha256=source_hash,
            final_render_sha256=pixel_sha256(candidate),
            attempts=attempts,
        ),
    )


def process_page(
    source: Image.Image,
    *,
    page_number: int,
    source_page_dpi: float,
    settings: Settings,
    client: ModelClient,
    finalize_candidate: Callable[[Image.Image], Image.Image],
) -> PageOutcome:
    original_source = source.convert("RGB")
    source_hash = pixel_sha256(original_source)
    source = original_source
    try:
        reading_rotation = client.reading_rotation(source)
    except GlobalProviderError:
        raise
    except (ReviewerResponseError, ProviderError):
        reading_rotation = 0
    if reading_rotation:
        source = rotate_reading_orientation(source, reading_rotation)
    page_plane = detect_page_plane(source)
    photographed_page = bool(
        page_plane is not None
        and page_plane.confidence >= 0.72
        and page_plane.area_fraction <= 0.98
    )
    attempts: list[AttemptRecord] = []
    base_prompt = PHOTO_RECTIFICATION_PROMPT if photographed_page else GENERATION_PROMPT
    prompt = base_prompt
    fallback_reason = "attempts_exhausted"
    cost_stopped = False
    if photographed_page:
        rectified_fallback: Image.Image | None = None
        for _geometry_attempt in range(settings.max_attempts):
            geometry_record = AttemptRecord(
                number=len(attempts) + 1,
                strategy="model_assisted_source_cleanup",
                effective_dpi=round(source_page_dpi, 2),
            )
            attempts.append(geometry_record)
            try:
                geometry = client.locate_page(source)
                rectified = (
                    rectify_page_geometry(source, geometry) if geometry is not None else None
                )
                if rectified is None:
                    geometry_record.local_issues = ["page_geometry_not_found"]
                    continue
                cleaned_reference = photographed_page_cleanup(rectified)
                rectified_fallback = rectified
                try:
                    generated = client.generate(
                        rectified,
                        prompt,
                        max_edge=settings.max_reference_edge,
                    )
                except PayloadTooLargeError:
                    generated = client.generate(
                        rectified,
                        prompt,
                        max_edge=max(1024, settings.max_reference_edge // 2),
                    )
                normalized = normalize_generated(
                    generated,
                    rectified.size,
                    source_dpi=source_page_dpi,
                )
                geometry_record.generated_width = normalized.generated_width
                geometry_record.generated_height = normalized.generated_height
                recreation = clear_page_border(finish_pristine_recreation(normalized.image))
                aligned_recreation = align_candidate_to_source(
                    rectified,
                    recreation,
                )
                if aligned_recreation is not None:
                    recreation = clear_page_border(aligned_recreation)
                candidate = finalize_candidate(recreation)
                deterministic = _validate_clean_candidate(
                    rectified,
                    candidate,
                    min_effective_dpi=settings.min_effective_dpi,
                    effective_dpi=source_page_dpi,
                )
                geometry_record.local_issues = deterministic.issues
                if not deterministic.accepted and not _photo_local_gate_accepts(
                    deterministic.issues
                ):
                    continue
                accepted, discrepancies = _verification_accepts(
                    client,
                    rectified,
                    candidate,
                    confirm_rejections=True,
                    quality_consensus=True,
                    register_views=True,
                    collect_all_views=True,
                )
                geometry_record.verification_categories = list(
                    dict.fromkeys(item.category for item in discrepancies)
                )
                geometry_record.accepted = accepted
                for _recovery_pass in range(_MAX_SOURCE_REGION_RECOVERY_PASSES):
                    source_regions = [
                        _source_region(item)
                        for item in discrepancies
                        if _preserve_from_source(item)
                    ]
                    if accepted or not source_regions:
                        break
                    recreation = replace_with_source_evidence_regions(
                        cleaned_reference,
                        recreation,
                        source_regions,
                        already_aligned=True,
                    )
                    candidate = finalize_candidate(recreation)
                    deterministic = _validate_clean_candidate(
                        rectified,
                        candidate,
                        min_effective_dpi=settings.min_effective_dpi,
                        effective_dpi=source_page_dpi,
                    )
                    geometry_record.local_issues = deterministic.issues
                    if not deterministic.accepted and not _photo_local_gate_accepts(
                        deterministic.issues
                    ):
                        accepted = False
                        break
                    accepted, discrepancies = _verification_accepts(
                        client,
                        rectified,
                        candidate,
                        confirm_rejections=True,
                        quality_consensus=True,
                        register_views=True,
                        collect_all_views=True,
                    )
                    geometry_record.verification_categories = list(
                        dict.fromkeys(item.category for item in discrepancies)
                    )
                    geometry_record.accepted = accepted
                if not accepted:
                    for _quality_repair in range(_MAX_PHOTO_QUALITY_REPAIRS):
                        quality_region = best_repair_region(
                            [
                                Discrepancy("unresolved_content", item.severity, item.region)
                                for item in discrepancies
                                if item.category == "scanner_quality"
                            ]
                        )
                        if quality_region is None:
                            break
                        recreation = repair_region(
                            rectified,
                            recreation,
                            quality_region,
                            client=client,
                            max_edge=settings.max_reference_edge,
                        )
                        candidate = finalize_candidate(recreation)
                        deterministic = _validate_clean_candidate(
                            rectified,
                            candidate,
                            min_effective_dpi=settings.min_effective_dpi,
                            effective_dpi=source_page_dpi,
                        )
                        geometry_record.local_issues = deterministic.issues
                        if not deterministic.accepted and not _photo_local_gate_accepts(
                            deterministic.issues
                        ):
                            break
                        accepted, discrepancies = _verification_accepts(
                            client,
                            rectified,
                            candidate,
                            confirm_rejections=True,
                            quality_consensus=True,
                            register_views=True,
                            collect_all_views=True,
                        )
                        geometry_record.verification_categories = list(
                            dict.fromkeys(item.category for item in discrepancies)
                        )
                        geometry_record.accepted = accepted
                        if accepted:
                            break
                if accepted:
                    return PageOutcome(
                        output_image=recreation,
                        record=PageRecord(
                            page=page_number,
                            status="model_assisted_clean",
                            source_render_sha256=source_hash,
                            final_render_sha256=pixel_sha256(candidate),
                            attempts=attempts,
                        ),
                    )
                prompt = base_prompt + _feedback(geometry_record.verification_categories)
            except GlobalProviderError:
                raise
            except (ContentPolicyError, ReviewerResponseError, ProviderError) as exc:
                geometry_record.error_type = _error_name(exc)
        if rectified_fallback is not None:
            photo_cleanup_record = AttemptRecord(
                number=len(attempts) + 1,
                strategy="source_preserving_cleanup",
                effective_dpi=round(source_page_dpi, 2),
            )
            attempts.append(photo_cleanup_record)
            try:
                recreation = photographed_page_cleanup(rectified_fallback)
                candidate = finalize_candidate(recreation)
                deterministic = _validate_clean_candidate(
                    rectified_fallback,
                    candidate,
                    min_effective_dpi=settings.min_effective_dpi,
                    effective_dpi=source_page_dpi,
                )
                photo_cleanup_record.local_issues = deterministic.issues
                if deterministic.accepted or _photo_local_gate_accepts(deterministic.issues):
                    accepted, discrepancies = _verification_accepts(
                        client,
                        rectified_fallback,
                        candidate,
                        confirm_rejections=True,
                        quality_consensus=True,
                        register_views=True,
                        collect_all_views=True,
                    )
                    evidence_regions = [
                        _source_region(item)
                        for item in discrepancies
                        if item.category in _SOURCE_EVIDENCE_RECOVERY_CATEGORIES
                    ]
                    if not accepted and evidence_regions:
                        recreation = restore_source_evidence_regions(
                            rectified_fallback,
                            recreation,
                            evidence_regions,
                            already_aligned=True,
                        )
                        candidate = finalize_candidate(recreation)
                        deterministic = _validate_clean_candidate(
                            rectified_fallback,
                            candidate,
                            min_effective_dpi=settings.min_effective_dpi,
                            effective_dpi=source_page_dpi,
                        )
                        photo_cleanup_record.local_issues = deterministic.issues
                        if deterministic.accepted or _photo_local_gate_accepts(
                            deterministic.issues
                        ):
                            accepted, discrepancies = _verification_accepts(
                                client,
                                rectified_fallback,
                                candidate,
                                confirm_rejections=True,
                                quality_consensus=True,
                                register_views=True,
                                collect_all_views=True,
                            )
                    if not accepted:
                        for _quality_repair in range(_MAX_PHOTO_QUALITY_REPAIRS):
                            quality_region = best_repair_region(
                                [
                                    Discrepancy(
                                        "unresolved_content",
                                        item.severity,
                                        item.region,
                                    )
                                    for item in discrepancies
                                    if item.category == "scanner_quality"
                                ]
                            )
                            if quality_region is None:
                                break
                            recreation = repair_region(
                                rectified_fallback,
                                recreation,
                                quality_region,
                                client=client,
                                max_edge=settings.max_reference_edge,
                            )
                            candidate = finalize_candidate(recreation)
                            deterministic = _validate_clean_candidate(
                                rectified_fallback,
                                candidate,
                                min_effective_dpi=settings.min_effective_dpi,
                                effective_dpi=source_page_dpi,
                            )
                            photo_cleanup_record.local_issues = deterministic.issues
                            if not deterministic.accepted and not _photo_local_gate_accepts(
                                deterministic.issues
                            ):
                                break
                            accepted, discrepancies = _verification_accepts(
                                client,
                                rectified_fallback,
                                candidate,
                                confirm_rejections=True,
                                quality_consensus=True,
                                register_views=True,
                                collect_all_views=True,
                            )
                            if accepted:
                                break
                    photo_cleanup_record.verification_categories = list(
                        dict.fromkeys(item.category for item in discrepancies)
                    )
                    photo_cleanup_record.accepted = accepted
                    if accepted:
                        return PageOutcome(
                            output_image=recreation,
                            record=PageRecord(
                                page=page_number,
                                status="source_preserving_clean",
                                source_render_sha256=source_hash,
                                final_render_sha256=pixel_sha256(candidate),
                                attempts=attempts,
                            ),
                        )
            except GlobalProviderError:
                raise
            except (ContentPolicyError, ReviewerResponseError, ProviderError) as exc:
                photo_cleanup_record.error_type = _error_name(exc)
        return PageOutcome(
            output_image=original_source,
            record=PageRecord(
                page=page_number,
                status="original_fallback",
                source_render_sha256=source_hash,
                final_render_sha256=source_hash,
                attempts=attempts,
                fallback_reason="attempts_exhausted",
            ),
        )
    if settings.backend == "agentbridge":
        try:
            source_first = _try_source_first_cleanup(
                source,
                page_number=page_number,
                source_hash=source_hash,
                source_page_dpi=source_page_dpi,
                settings=settings,
                client=client,
                finalize_candidate=finalize_candidate,
                attempts=attempts,
            )
            if source_first is not None:
                return source_first
        except GlobalProviderError:
            raise
        except (ContentPolicyError, ReviewerResponseError, ProviderError):
            # A page-scoped review failure must not prevent the established
            # generative path from attempting the page.
            pass
    for _attempt_number in range(1, settings.max_attempts + 1):
        record = AttemptRecord(number=len(attempts) + 1)
        attempts.append(record)
        max_edge = settings.max_reference_edge
        try:
            try:
                generated = client.generate(source, prompt, max_edge=max_edge)
            except PayloadTooLargeError:
                generated = client.generate(source, prompt, max_edge=max(1024, max_edge // 2))
            normalized = normalize_generated(
                generated,
                source.size,
                source_dpi=source_page_dpi,
            )
            record.generated_width = normalized.generated_width
            record.generated_height = normalized.generated_height
            generated_reference = clear_page_border(finish_pristine_recreation(normalized.image))
            if photographed_page:
                rectified_source = rectify_source_to_reference(source, generated_reference)
                if rectified_source is None:
                    record.local_issues = ["page_registration_failed"]
                    prompt = base_prompt + _feedback(["unresolved_content"])
                    continue
                record.strategy = "model_assisted_source_cleanup"
                recreation = source_preserving_cleanup(rectified_source)
            else:
                recreation = rescue_colored_marks(source, generated_reference)
            record.effective_dpi = round(source_page_dpi, 2)
            candidate = finalize_candidate(recreation)
            deterministic = _validate_clean_candidate(
                source,
                candidate,
                min_effective_dpi=settings.min_effective_dpi,
                effective_dpi=source_page_dpi,
            )
            record.local_issues = deterministic.issues
            # A full-page recreation can solve paper tone, folds, and scanner
            # noise while still faithfully reproducing physical punch holes.
            # Do not discard that otherwise useful recreation and ask the model
            # to regenerate the whole page again. Apply the same bounded,
            # context-aware authored-hole repair used by the source-first path,
            # then run every deterministic and semantic gate on the combined
            # result. This is intentionally limited to scans: photographed pages
            # have a separate rectification and regional-repair workflow above.
            hole_regions = authored_punch_hole_regions(source)[:_MAX_AUTHORED_HOLE_REPAIRS]
            if (
                not photographed_page
                and hole_regions
                and set(deterministic.issues) == {"residual_punch_hole"}
            ):
                recreation, candidate, deterministic = _repair_authored_hole_regions(
                    source,
                    recreation,
                    candidate,
                    hole_regions,
                    settings=settings,
                    client=client,
                    finalize_candidate=finalize_candidate,
                    source_page_dpi=source_page_dpi,
                    prefer_deterministic_erase=True,
                )
                record.local_issues = deterministic.issues
            if not deterministic.accepted:
                prompt = base_prompt + _feedback(["unresolved_content"])
                continue
            accepted, discrepancies = _verification_accepts(client, source, candidate)
            if photographed_page:
                categories = list(dict.fromkeys(item.category for item in discrepancies))
                record.verification_categories = categories
                record.accepted = accepted
                if accepted:
                    return PageOutcome(
                        output_image=recreation,
                        record=PageRecord(
                            page=page_number,
                            status="model_assisted_clean",
                            source_render_sha256=source_hash,
                            final_render_sha256=pixel_sha256(candidate),
                            attempts=attempts,
                        ),
                    )
                prompt = base_prompt + _feedback(categories)
                continue
            if not accepted:
                source_regions = [
                    _source_region(item) for item in discrepancies if _preserve_from_source(item)
                ]
                if source_regions:
                    recreation = restore_source_regions(source, recreation, source_regions)
                model_discrepancies = [
                    item
                    for item in discrepancies
                    if not _preserve_from_source(item) and item.category != "scanner_quality"
                ]
                repair_box = best_repair_region(model_discrepancies)
                if repair_box is not None:
                    recreation = repair_region(
                        source,
                        recreation,
                        repair_box,
                        client=client,
                        max_edge=settings.max_reference_edge,
                    )
                candidate = finalize_candidate(recreation)
                deterministic = _validate_clean_candidate(
                    source,
                    candidate,
                    min_effective_dpi=settings.min_effective_dpi,
                    effective_dpi=source_page_dpi,
                )
                record.local_issues = deterministic.issues
                if deterministic.accepted and (source_regions or repair_box is not None):
                    accepted, discrepancies = _verification_accepts(client, source, candidate)
            categories = list(dict.fromkeys(item.category for item in discrepancies))
            record.verification_categories = categories
            record.accepted = accepted
            if accepted:
                return PageOutcome(
                    output_image=recreation,
                    record=PageRecord(
                        page=page_number,
                        status="model_generated_clean",
                        source_render_sha256=source_hash,
                        final_render_sha256=pixel_sha256(candidate),
                        attempts=attempts,
                    ),
                )
            prompt = base_prompt + _feedback(categories)
        except ContentPolicyError as exc:
            record.error_type = _error_name(exc)
            fallback_reason = "content_policy"
            break
        except CostLimitReached as exc:
            record.error_type = _error_name(exc)
            fallback_reason = "cost_limit"
            cost_stopped = True
            break
        except GlobalProviderError:
            raise
        except (ReviewerResponseError, ProviderError) as exc:
            record.error_type = _error_name(exc)
            fallback_reason = "provider_or_review_error"
            # Page-scoped provider failures consume the attempt; auth/config errors
            # have already been normalized as GlobalProviderError and propagate.
            continue
    if (
        settings.backend != "agentbridge"
        and not photographed_page
        and fallback_reason in {"attempts_exhausted", "provider_or_review_error"}
    ):
        try:
            source_retry = _try_source_first_cleanup(
                source,
                page_number=page_number,
                source_hash=source_hash,
                source_page_dpi=source_page_dpi,
                settings=settings,
                client=client,
                finalize_candidate=finalize_candidate,
                attempts=attempts,
            )
            if source_retry is not None:
                return source_retry
        except ContentPolicyError as exc:
            attempts[-1].error_type = _error_name(exc)
            fallback_reason = "content_policy"
        except CostLimitReached as exc:
            attempts[-1].error_type = _error_name(exc)
            fallback_reason = "cost_limit"
            cost_stopped = True
        except GlobalProviderError:
            raise
        except (ReviewerResponseError, ProviderError) as exc:
            attempts[-1].error_type = _error_name(exc)
            fallback_reason = "provider_or_review_error"
    return PageOutcome(
        output_image=original_source,
        record=PageRecord(
            page=page_number,
            status="original_fallback",
            source_render_sha256=source_hash,
            final_render_sha256=source_hash,
            attempts=attempts,
            fallback_reason=fallback_reason,
        ),
        cost_stopped=cost_stopped,
    )


def _core_manifest(report: DocumentReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "source_sha256": report.source_sha256,
        "backend": report.backend,
        "billing_mode": report.billing_mode,
        "models": {"image": report.image_model, "verification": report.verification_model},
        "verification": {"strategy": report.verification_strategy},
        "pages": [
            {"page": page.page, "status": page.status, "fallback_reason": page.fallback_reason}
            for page in report.pages
        ],
    }


def _base_report(paths: OutputPaths, settings: Settings) -> DocumentReport:
    return DocumentReport(
        schema_version=7,
        run_id=uuid4().hex,
        source=str(paths.source),
        output=str(paths.output),
        source_sha256=sha256_file(paths.source),
        output_sha256=None,
        backend=settings.backend,
        billing_mode=(
            "codex_subscription" if settings.backend == "agentbridge" else "openrouter_usd"
        ),
        image_model=settings.image_model,
        verification_model=settings.review_model,
        verification_strategy=VERIFICATION_STRATEGY,
        started_at=_now(),
    )


def _finish_report(report: DocumentReport, client: ModelClient) -> None:
    report.finished_at = _now()
    report.backend_version = getattr(client, "backend_version", None)
    report.cost_usd = float(client.costs.total) if report.backend == "openrouter" else None
    report.prompt_tokens = client.costs.prompt_tokens
    report.completion_tokens = client.costs.completion_tokens
    report.total_tokens = client.costs.total_tokens
    report.ambiguous_timeout_charges = client.costs.ambiguous_timeouts


def clean_image(
    paths: OutputPaths,
    settings: Settings,
    client: ModelClient,
    *,
    force: bool,
) -> DocumentReport:
    report = _base_report(paths, settings)
    source_bytes = paths.source.read_bytes()
    image = load_image(paths.source)
    suffix = paths.output.suffix.lower()

    def finalize(candidate: Image.Image) -> Image.Image:
        return final_pixel_image(candidate, suffix)[1]

    outcome = process_page(
        image,
        page_number=1,
        source_page_dpi=source_dpi(image),
        settings=settings,
        client=client,
        finalize_candidate=finalize,
    )
    report.pages.append(outcome.record)
    wrapper = manifest_wrapper(_core_manifest(report))
    if outcome.record.status == "original_fallback":
        published = embed_image(source_bytes, paths.source.suffix, wrapper)
    else:
        encoded, exact = final_pixel_image(outcome.output_image, suffix)
        outcome.record.final_render_sha256 = pixel_sha256(exact)
        published = embed_image(encoded, suffix, wrapper)
    staged_output = staged_path(paths.output)
    staged_report = staged_path(paths.report)
    try:
        private_write(staged_output, published)
        report.output_sha256 = sha256_file(staged_output)
        _finish_report(report, client)
        write_report(staged_report, report.as_dict())
        publish_pair(staged_output, staged_report, paths.output, paths.report, force=force)
    finally:
        staged_output.unlink(missing_ok=True)
        staged_report.unlink(missing_ok=True)
    return report


def clean_pdf(
    paths: OutputPaths,
    settings: Settings,
    client: ModelClient,
    *,
    force: bool,
) -> DocumentReport:
    inspection = inspect_pdf(paths.source)
    originals = render_pages(paths.source, dpi=settings.render_dpi)
    report = _base_report(paths, settings)
    report.removed_pdf_features = inspection.removed_features
    report.warnings = inspection.warnings
    output_images: list[Image.Image] = []
    cost_stopped = False
    with private_workdir() as directory:
        for index, page in enumerate(originals):
            if cost_stopped:
                outcome = PageOutcome(
                    output_image=page.image,
                    record=PageRecord(
                        page=index + 1,
                        status="original_fallback",
                        source_render_sha256=pixel_sha256(page.image),
                        final_render_sha256=pixel_sha256(page.image),
                        fallback_reason="cost_limit",
                    ),
                    cost_stopped=True,
                )
            else:

                def finalize(candidate: Image.Image, page_index: int = index) -> Image.Image:
                    preview = directory / f"preview-{page_index}.pdf"
                    return render_overlay_preview(
                        paths.source,
                        page_index,
                        candidate,
                        preview,
                        dpi=settings.render_dpi,
                    )

                outcome = process_page(
                    page.image,
                    page_number=index + 1,
                    source_page_dpi=page.dpi,
                    settings=settings,
                    client=client,
                    finalize_candidate=finalize,
                )
            report.pages.append(outcome.record)
            output_images.append(outcome.output_image)
            cost_stopped = cost_stopped or outcome.cost_stopped

        wrapper = manifest_wrapper(_core_manifest(report))
        staged_output = staged_path(paths.output)
        staged_report = staged_path(paths.report)
        try:
            build_pdf(
                paths.source,
                staged_output,
                output_images,
                manifest=wrapper,
                run_id=report.run_id,
            )
            final_pages = render_pages(staged_output, dpi=settings.render_dpi)
            if len(final_pages) != len(originals):
                raise ValueError("published PDF page count changed")
            for original, final, record in zip(originals, final_pages, report.pages, strict=True):
                if original.text_signature != final.text_signature:
                    raise ValueError("published PDF searchable text layer changed")
                record.final_render_sha256 = pixel_sha256(final.image)
            report.output_sha256 = sha256_file(staged_output)
            _finish_report(report, client)
            write_report(staged_report, report.as_dict())
            publish_pair(staged_output, staged_report, paths.output, paths.report, force=force)
        finally:
            staged_output.unlink(missing_ok=True)
            staged_report.unlink(missing_ok=True)
    return report


def clean_document(
    paths: OutputPaths,
    settings: Settings,
    client: ModelClient,
    *,
    force: bool,
) -> DocumentReport:
    if paths.source.suffix.lower() == ".pdf":
        return clean_pdf(paths, settings, client, force=force)
    return clean_image(paths, settings, client, force=force)


def report_has_fallback(report: DocumentReport) -> bool:
    return any(page.status == "original_fallback" for page in report.pages)


def report_summary(report: DocumentReport) -> str:
    generated = sum(page.status != "original_fallback" for page in report.pages)
    fallback = len(report.pages) - generated
    return json.dumps(
        {"output": report.output, "generated_pages": generated, "fallback_pages": fallback},
        separators=(",", ":"),
    )
