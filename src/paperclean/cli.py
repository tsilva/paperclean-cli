"""Command-line interface for PaperClean."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from paperclean import __version__
from paperclean.config import Settings
from paperclean.discovery import OutputPaths, check_collision, discover, output_paths
from paperclean.errors import (
    ConfigurationError,
    GlobalOpenRouterError,
    InputError,
    OutputCollisionError,
    PaperCleanError,
)
from paperclean.openrouter import OpenRouterClient
from paperclean.pdfs import page_count
from paperclean.pipeline import clean_document, report_has_fallback, report_summary
from paperclean.validation import check_tesseract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperclean",
        description="Clean phone-scanned PDFs and images with conservative fidelity review.",
    )
    parser.add_argument("input", type=Path, help="PDF/image file or recursively scanned directory")
    parser.add_argument("--output", type=Path, help="single-file output override")
    parser.add_argument(
        "--force", action="store_true", help="atomically replace existing output/report"
    )
    parser.add_argument("--jobs", type=int, help="documents processed concurrently (default: 1)")
    parser.add_argument(
        "--max-attempts", type=int, help="generation attempts per page (default: 3)"
    )
    parser.add_argument("--image-model", help="OpenRouter image generation model")
    parser.add_argument("--review-model", help="OpenRouter multimodal review model")
    parser.add_argument("--ocr-lang", help="Tesseract language(s), such as eng or eng+por")
    parser.add_argument("--max-cost-usd", help="soft observed OpenRouter cost ceiling")
    parser.add_argument("--zdr", action="store_true", default=None, help="require ZDR endpoints")
    parser.add_argument("--yes", action="store_true", help="confirm estimated semantic call count")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_sources(
        {
            "jobs": args.jobs,
            "max_attempts": args.max_attempts,
            "image_model": args.image_model,
            "review_model": args.review_model,
            "ocr_lang": args.ocr_lang,
            "max_cost_usd": args.max_cost_usd,
            "zdr": args.zdr,
        }
    )


def _prepare_paths(
    sources: list[Path],
    *,
    output: Path | None,
    force: bool,
    directory_input: bool,
) -> tuple[list[OutputPaths], list[OutputPaths]]:
    if output is not None and (len(sources) != 1 or directory_input):
        raise InputError("--output can be used only when INPUT is one file")
    ready: list[OutputPaths] = []
    skipped: list[OutputPaths] = []
    for source in sources:
        paths = output_paths(source, output)
        try:
            check_collision(paths, force=force)
        except OutputCollisionError:
            if directory_input and not force:
                skipped.append(paths)
                continue
            raise
        ready.append(paths)
    return ready, skipped


def _count_pages(paths: list[OutputPaths]) -> int:
    total = 0
    for item in paths:
        total += page_count(item.source) if item.source.suffix.lower() == ".pdf" else 1
    return total


def _confirm(*, page_total: int, document_total: int, max_attempts: int, yes: bool) -> None:
    maximum = page_total * max_attempts * 6
    print(
        f"Preflight: {document_total} document(s), {page_total} page(s), "
        f"up to {maximum} semantic model calls."
    )
    if yes or (page_total == 1 and document_total == 1):
        return
    if not sys.stdin.isatty():
        raise ConfigurationError("confirmation is required; rerun with --yes")
    response = input("Continue? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        raise ConfigurationError("cancelled")


def _process_documents(
    paths: list[OutputPaths],
    settings: Settings,
    client: OpenRouterClient,
    *,
    force: bool,
) -> tuple[list[object], BaseException | None]:
    reports: list[object | None] = [None] * len(paths)
    fatal: BaseException | None = None
    stop = threading.Event()
    cursor = 0
    cursor_lock = threading.Lock()

    def worker() -> None:
        nonlocal cursor, fatal
        while not stop.is_set():
            with cursor_lock:
                if cursor >= len(paths):
                    return
                index = cursor
                cursor += 1
            try:
                reports[index] = clean_document(paths[index], settings, client, force=force)
            except GlobalOpenRouterError as exc:
                with cursor_lock:
                    if fatal is None:
                        fatal = exc
                stop.set()
                return
            except BaseException as exc:
                with cursor_lock:
                    if fatal is None:
                        fatal = exc
                stop.set()
                return

    jobs = min(settings.paid_jobs, max(1, len(paths)))
    if jobs == 1:
        worker()
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="paperclean") as executor:
            futures = [executor.submit(worker) for _ in range(jobs)]
            for future in as_completed(futures):
                future.result()
    return [report for report in reports if report is not None], fatal


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sources = discover(args.input)
        if not sources:
            raise InputError(f"no supported PDF, JPEG, or PNG files found under {args.input}")
        directory_input = args.input.expanduser().resolve().is_dir()
        paths, skipped = _prepare_paths(
            sources,
            output=args.output,
            force=args.force,
            directory_input=directory_input,
        )
        if skipped:
            for item in skipped:
                print(f"skip existing: {item.output}", file=sys.stderr)
        if not paths:
            return 0
        settings = _settings(args)
        check_tesseract(settings.ocr_lang)
        pages = _count_pages(paths)
        _confirm(
            page_total=pages,
            document_total=len(paths),
            max_attempts=settings.max_attempts,
            yes=args.yes,
        )
        with OpenRouterClient(settings) as client:
            client.preflight()
            reports, fatal = _process_documents(paths, settings, client, force=args.force)
        for report in reports:
            print(report_summary(report))  # type: ignore[arg-type]
        if fatal is not None:
            print(f"paperclean: {fatal}", file=sys.stderr)
            return 1
        return 2 if any(report_has_fallback(report) for report in reports) else 0  # type: ignore[arg-type]
    except KeyboardInterrupt:
        print("paperclean: interrupted", file=sys.stderr)
        return 1
    except PaperCleanError as exc:
        print(f"paperclean: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"paperclean: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())
