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
    source_dpi,
)
from paperclean.models import AttemptRecord, Discrepancy, DocumentReport, PageRecord
from paperclean.pdfs import build_pdf, inspect_pdf, render_overlay_preview, render_pages
from paperclean.prompting import FEEDBACK_TEMPLATE, GENERATION_PROMPT, load_prompt
from paperclean.provenance import embed_image, manifest_wrapper, write_report
from paperclean.providers import ModelClient
from paperclean.restoration import (
    best_repair_region,
    clear_page_border,
    registered_review_pairs,
    repair_region,
    rescue_colored_marks,
    rescue_edge_text,
    restore_source_regions,
)
from paperclean.util import (
    private_workdir,
    private_write,
    publish_pair,
    sha256_file,
    staged_path,
)
from paperclean.validation import validate_candidate


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


def _reviews_accept(
    client: ModelClient,
    source: Image.Image,
    candidate: Image.Image,
) -> tuple[bool, list[Discrepancy]]:
    discrepancies: list[Discrepancy] = []
    for index, (source_view, candidate_view) in enumerate(
        registered_review_pairs(source, candidate)
    ):
        verdict = None
        for schema_attempt in range(2):
            try:
                verdict = client.review(
                    source_view,
                    candidate_view,
                    view_name="full page" if index == 0 else f"region {index} of 4",
                )
                break
            except ReviewerResponseError:
                if schema_attempt:
                    raise
        if verdict is None:
            raise ReviewerResponseError("reviewer did not produce a verdict")
        view_box = (
            (0, 0, candidate.width, candidate.height)
            if index == 0
            else review_boxes(candidate.size)[index - 1]
        )
        view_left, view_top, view_right, view_bottom = view_box
        view_width = view_right - view_left
        view_height = view_bottom - view_top
        for item in verdict.discrepancies:
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
        if not verdict.accepted:
            if not verdict.content_match and not discrepancies:
                discrepancies.append(
                    Discrepancy("unresolved_content", "high", (0.0, 0.0, 1.0, 1.0))
                )
            if not verdict.scanner_quality:
                discrepancies.append(Discrepancy("scanner_quality", "high", (0.0, 0.0, 1.0, 1.0)))
            return False, discrepancies
    return True, discrepancies


_SOURCE_PRESERVED_CATEGORIES = {
    "changed_handwriting",
    "changed_signature",
    "changed_stamp",
    "changed_redaction",
}
_SOURCE_EDGE_CATEGORIES = {
    "changed_text",
    "missing_text",
    "invented_text",
    "cropped_content",
    "unresolved_content",
}


def _preserve_from_source(discrepancy: Discrepancy) -> bool:
    if discrepancy.category in _SOURCE_PRESERVED_CATEGORIES:
        return True
    if discrepancy.category not in _SOURCE_EDGE_CATEGORIES:
        return False
    left, top, right, bottom = discrepancy.region
    return min(left, top) < 0.04 or max(right, bottom) > 0.94


def _source_region(discrepancy: Discrepancy) -> tuple[float, float, float, float]:
    left, top, right, bottom = discrepancy.region
    if bottom > 0.94:
        return (0.0, min(top, 0.94), 1.0, 1.0)
    return (left, top, right, bottom)


def process_page(
    source: Image.Image,
    *,
    page_number: int,
    source_page_dpi: float,
    settings: Settings,
    client: ModelClient,
    finalize_candidate: Callable[[Image.Image], Image.Image],
) -> PageOutcome:
    source = source.convert("RGB")
    source_hash = pixel_sha256(source)
    attempts: list[AttemptRecord] = []
    prompt = GENERATION_PROMPT
    fallback_reason = "attempts_exhausted"
    cost_stopped = False
    for attempt_number in range(1, settings.max_attempts + 1):
        record = AttemptRecord(number=attempt_number)
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
            recreation = rescue_colored_marks(
                source,
                rescue_edge_text(
                    source,
                    clear_page_border(finish_pristine_recreation(normalized.image)),
                    language=settings.ocr_lang,
                ),
            )
            record.effective_dpi = round(source_page_dpi, 2)
            candidate = finalize_candidate(recreation)
            deterministic = validate_candidate(
                source,
                candidate,
                language=settings.ocr_lang,
                min_effective_dpi=settings.min_effective_dpi,
                effective_dpi=source_page_dpi,
            )
            record.deterministic_issues = deterministic.issues
            if not deterministic.accepted:
                prompt = GENERATION_PROMPT + _feedback(["unresolved_content"])
                continue
            if not settings.review_enabled:
                record.accepted = True
                return PageOutcome(
                    output_image=recreation,
                    record=PageRecord(
                        page=page_number,
                        status="model_generated_unreviewed",
                        source_render_sha256=source_hash,
                        final_render_sha256=pixel_sha256(candidate),
                        attempts=attempts,
                    ),
                )
            accepted, discrepancies = _reviews_accept(client, source, candidate)
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
                deterministic = validate_candidate(
                    source,
                    candidate,
                    language=settings.ocr_lang,
                    min_effective_dpi=settings.min_effective_dpi,
                    effective_dpi=source_page_dpi,
                )
                record.deterministic_issues = deterministic.issues
                if deterministic.accepted and (source_regions or repair_box is not None):
                    accepted, discrepancies = _reviews_accept(client, source, candidate)
            categories = list(dict.fromkeys(item.category for item in discrepancies))
            record.review_categories = categories
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
            prompt = GENERATION_PROMPT + _feedback(categories)
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
    return PageOutcome(
        output_image=source,
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
        "review_enabled": report.review_enabled,
        "models": {"image": report.image_model, "review": report.review_model},
        "pages": [
            {"page": page.page, "status": page.status, "fallback_reason": page.fallback_reason}
            for page in report.pages
        ],
    }


def _base_report(paths: OutputPaths, settings: Settings) -> DocumentReport:
    return DocumentReport(
        schema_version=3,
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
        review_enabled=settings.review_enabled,
        review_model=settings.review_model if settings.review_enabled else None,
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
                    raise ValueError("published PDF OCR text layer changed")
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
