"""Command-line interface for PaperClean."""

from __future__ import annotations

import argparse
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import cast

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from paperclean import __version__
from paperclean.agentbridge import AgentBridgeClient
from paperclean.config import Settings
from paperclean.discovery import OutputPaths, check_collision, discover, output_paths
from paperclean.environment import (
    discover_runtime_environment,
    relaunch_with_keyenv,
    restore_keyenv_working_directory,
)
from paperclean.errors import (
    ConfigurationError,
    GlobalProviderError,
    InputError,
    OutputCollisionError,
    PaperCleanError,
)
from paperclean.openrouter import OpenRouterClient
from paperclean.pdfs import page_count
from paperclean.pipeline import clean_document, report_has_fallback, report_summary
from paperclean.preflight import CostProjection
from paperclean.providers import ModelClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperclean",
        description="Clean phone-scanned PDFs and images with mandatory model verification.",
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
    parser.add_argument(
        "--backend",
        choices=("openrouter", "agentbridge"),
        help="model backend (default: openrouter)",
    )
    parser.add_argument(
        "--agentbridge-base-url",
        help="loopback AgentBridge API base URL",
    )
    parser.add_argument(
        "--agentbridge-timeout",
        type=int,
        help="AgentBridge request timeout in seconds",
    )
    parser.add_argument("--image-model", help="image generation or Codex orchestrator model")
    parser.add_argument("--review-model", help="multimodal verification model")
    parser.add_argument("--max-cost-usd", help="soft observed OpenRouter cost ceiling")
    parser.add_argument("--zdr", action="store_true", default=None, help="require ZDR endpoints")
    parser.add_argument(
        "--yes", action="store_true", help="accept the cost preflight without prompting"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _settings(args: argparse.Namespace, relaunch_args: Sequence[str]) -> Settings:
    runtime = discover_runtime_environment()
    backend = (
        str(args.backend or runtime.values.get("PAPERCLEAN_BACKEND", "openrouter")).strip().lower()
    )
    if args.agentbridge_base_url is not None and backend != "agentbridge":
        raise ConfigurationError("--agentbridge-base-url requires --backend agentbridge")
    if runtime.keyenv_manifest is not None and backend == "openrouter":
        relaunch_with_keyenv(runtime, relaunch_args)
    return Settings.from_sources(
        {
            "backend": args.backend,
            "base_url": args.agentbridge_base_url,
            "agentbridge_timeout": args.agentbridge_timeout,
            "jobs": args.jobs,
            "max_attempts": args.max_attempts,
            "image_model": args.image_model,
            "review_model": args.review_model,
            "max_cost_usd": args.max_cost_usd,
            "zdr": args.zdr,
        },
        runtime.values,
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


def _money(value: Decimal | None) -> str:
    return "unavailable" if value is None else f"${value:.4f}"


def _confirm(projection: CostProjection, *, yes: bool) -> None:
    console = Console()
    scenarios = (
        projection.one_pass,
        projection.configured_max,
        projection.recovery_ceiling,
    )
    available = projection.effective_available_usd
    required = projection.recovery_ceiling.cost_usd
    insufficient = available is not None and required is not None and available < required

    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="bold cyan", no_wrap=True)
    facts.add_column()
    facts.add_column(style="bold cyan", no_wrap=True)
    facts.add_column()
    facts.add_row(
        "Documents",
        str(projection.document_total),
        "Pages",
        str(projection.page_total),
    )
    facts.add_row("Max attempts/page", str(projection.max_attempts), "", "")

    models = Table.grid(padding=(0, 1))
    models.add_column(style="dim", no_wrap=True)
    models.add_column(overflow="fold")
    models.add_row(
        "Generation",
        Text(f"{projection.image_model}  ·  {projection.image_provider}"),
    )
    models.add_row(
        "Verification",
        Text(f"{projection.review_model}  ·  {projection.review_provider}"),
    )

    work = Table(box=box.ROUNDED, expand=True, header_style="bold white")
    work.add_column("Work", style="cyan", no_wrap=True)
    work.add_column("One pass", justify="right", style="green")
    work.add_column("Configured max", justify="right", style="yellow")
    work.add_column("Recovery ceiling", justify="right", style="magenta")
    work.add_row("Image generations", *(str(item.generations) for item in scenarios))
    work.add_row("Fidelity verifications", *(str(item.reviews) for item in scenarios))
    work.add_row(
        "Paid model calls" if projection.billing_mode == "openrouter_usd" else "Model calls",
        *(str(item.paid_calls) for item in scenarios),
    )
    work.add_row(
        "Projected cost",
        *(_money(item.cost_usd) for item in scenarios),
        style="bold",
    )

    balances = Table.grid(padding=(0, 2))
    balances.add_column(style="dim", no_wrap=True)
    balances.add_column(justify="right")
    if projection.billing_mode == "openrouter_usd":
        key_balance = (
            "unlimited" if projection.key_unlimited else _money(projection.key_remaining_usd)
        )
        balances.add_row("OpenRouter account", _money(projection.account_remaining_usd))
        balances.add_row("Key limit remaining", key_balance)
        balances.add_row("Soft cost limit", _money(projection.soft_limit_usd))
        note = Text(
            "Conservative projection · actual cost varies with image shape, tokens, "
            "early rejection, provider pricing, and retries.",
            style="dim",
        )
    else:
        balances.add_row("Billing", "Codex subscription")
        balances.add_row("USD cost", "unavailable")
        balances.add_row("AgentBridge", projection.backend_version or "unknown version")
        note = Text(
            "AgentBridge reports Codex orchestration tokens, not image-generation "
            "subscription consumption or a USD balance.",
            style="dim",
        )
    status = (
        Text.assemble(
            ("INSUFFICIENT CREDITS  ", "bold white on red"),
            (
                f"  {_money(available)} available; "
                f"{_money(required)} required for the recovery ceiling.",
                "bold red",
            ),
        )
        if insufficient
        else Text("READY FOR CONFIRMATION", style="bold green")
    )
    soft_limit_warning = (
        projection.billing_mode == "openrouter_usd"
        and projection.one_pass.cost_usd is not None
        and projection.soft_limit_usd is not None
        and projection.soft_limit_usd < projection.one_pass.cost_usd
    )
    contents: list[RenderableType] = [facts, Text(), models]
    contents.extend([Text(), work, Text(), balances, Text(), note])
    if soft_limit_warning:
        contents.extend(
            [
                Text(),
                Text(
                    f"Soft limit {_money(projection.soft_limit_usd)} is below the "
                    f"one-pass projection {_money(projection.one_pass.cost_usd)}.",
                    style="bold yellow",
                ),
            ]
        )
    contents.extend([Text(), status])
    console.print(
        Panel(
            Group(*contents),
            title=(
                "[bold cyan]PaperClean[/] · Paid-work preflight"
                if projection.billing_mode == "openrouter_usd"
                else "[bold cyan]PaperClean[/] · Codex-work preflight"
            ),
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if insufficient:
        raise ConfigurationError(
            "insufficient OpenRouter credits for the conservative recovery-ceiling "
            "projection; add credits before running"
        )
    if yes:
        return
    if not sys.stdin.isatty():
        raise ConfigurationError("confirmation is required; rerun with --yes")
    question = (
        "Begin paid model calls?"
        if projection.billing_mode == "openrouter_usd"
        else "Begin Codex model calls?"
    )
    if not Confirm.ask(question, default=False, console=console):
        raise ConfigurationError("cancelled")


def _process_documents(
    paths: list[OutputPaths],
    settings: Settings,
    client: ModelClient,
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
            except GlobalProviderError as exc:
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
    relaunch_args = list(sys.argv[1:] if argv is None else argv)
    try:
        restore_keyenv_working_directory()
        args = build_parser().parse_args(argv)
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
        settings = _settings(args, relaunch_args)
        pages = _count_pages(paths)
        raw_client = cast(
            ModelClient,
            AgentBridgeClient(settings)
            if settings.backend == "agentbridge"
            else OpenRouterClient(settings),
        )
        with raw_client as client:
            client.preflight()
            projection = client.cost_projection(
                page_total=pages,
                document_total=len(paths),
                max_attempts=settings.max_attempts,
            )
            _confirm(projection, yes=args.yes)
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
