"""Document snapshots and user-directed refinement for the local workspace."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from PIL import Image

from paperclean.pdfs import build_pdf

if TYPE_CHECKING:
    from paperclean.lab import Lab


def feedback_for(run: dict[str, Any]) -> dict[str, Any]:
    issues = (run.get("review") or {}).get("issues", [])
    judgments, comments = run.get("judgments", {}), run.get("comments", {})
    accepted, dismissed, annotated = [], [], []
    for issue in issues:
        key = str(issue["id"])
        if judgments.get(key) == "confirmed":
            accepted.append(issue)
        if judgments.get(key) == "false_alarm":
            dismissed.append({"id": issue["id"], "label": issue["label"]})
        if comments.get(key, "").strip():
            annotated.append(
                {
                    "issue": issue,
                    "comment": comments[key],
                    "decision": judgments.get(key, "unrated"),
                }
            )
    return {
        "accepted_issues": accepted,
        "dismissed_issues": dismissed,
        "issue_comments": annotated,
        "page_comment": run.get("notes", ""),
    }


def has_repairs(feedback: dict[str, Any]) -> bool:
    return bool(
        feedback["accepted_issues"]
        or feedback["issue_comments"]
        or feedback["page_comment"].strip()
    )


class Workspace:
    def __init__(self, lab: Lab) -> None:
        self.lab = lab
        self.directory = lab.root / "versions"
        self.directory.mkdir(exist_ok=True)

    def _versions(self, document_id: str) -> list[dict[str, Any]]:
        versions = [json.loads(p.read_text()) for p in self.directory.glob("*/version.json")]
        return sorted(
            (v for v in versions if v["document_id"] == document_id), key=lambda v: v["number"]
        )

    def legacy(self, document_id: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
        pages = {}
        versions = self._versions(document_id)
        if versions:
            runs = [r for r in runs if r["created_at"] < versions[0]["created_at"]]
        for run in sorted(runs, key=lambda r: r["created_at"]):
            if (
                run.get("document_id") == document_id
                and run.get("restored_url")
                and not run.get("notes", "").startswith("Invalid source/candidate pairing")
            ):
                pages[str(run["page"])] = run["id"]
        return {
            "id": "legacy",
            "document_id": document_id,
            "number": 0,
            "name": "Existing restorations",
            "created_at": "",
            "parent_id": None,
            "page_runs": pages,
            "changed_pages": [],
            "mode": "legacy",
        }

    def versions(
        self, documents: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        result = []
        by_id = {r["id"]: r for r in runs}
        for document in documents:
            versions = self._versions(document["id"])
            legacy = self.legacy(document["id"], runs)
            if legacy["page_runs"]:
                versions = [legacy, *versions]
            for version in versions:
                changed = [
                    by_id[i]
                    for p, i in version["page_runs"].items()
                    if int(p) in version["changed_pages"]
                ]
                statuses = [r["status"] for r in changed]
                version["status"] = (
                    "processing"
                    if any(s in {"queued", "restoring", "reviewing"} for s in statuses)
                    else "needs_attention"
                    if any(s != "complete" for s in statuses)
                    else "complete"
                )
                version["restored_pages"] = sum(
                    bool(
                        self.candidate_path(
                            by_id.get(version["page_runs"].get(str(p["number"]), ""))
                        )
                    )
                    for p in document["pages"]
                )
                version["page_count"] = len(document["pages"])
                if (
                    version["status"] == "complete"
                    and version["restored_pages"] < version["page_count"]
                ):
                    version["status"] = "partial"
                result.append(version)
        return result

    def candidate_path(self, run: dict[str, Any] | None) -> Path | None:
        if not run:
            return None
        path: Path = self.lab.root / "runs" / run["id"] / "restored.png"
        if path.is_file():
            return path
        if run.get("parent_run_id"):
            parent = json.loads(
                (self.lab.root / "runs" / run["parent_run_id"] / "run.json").read_text()
            )
            return self.candidate_path(parent)
        return None

    def reviewed_run(self, run: dict[str, Any] | None) -> dict[str, Any] | None:
        if run and run["status"] == "complete":
            return run
        if (
            run
            and run["status"] in {"failed", "interrupted"}
            and not run.get("restored_url")
            and run.get("parent_run_id")
        ):
            parent = json.loads(
                (self.lab.root / "runs" / run["parent_run_id"] / "run.json").read_text()
            )
            return self.reviewed_run(parent)
        return None

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("approved") is not True:
            raise ValueError("Confirm transmission of page images before running")
        ident = body.get("document_id", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[a-f0-9]{32}", ident):
            raise ValueError("Select a document")
        mode = body.get("mode", "restore")
        if mode not in {"restore", "refine"}:
            raise ValueError("Invalid document operation")
        with self.lab.lock:
            state = self.lab.state()
            document = next((d for d in state["documents"] if d["id"] == ident), None)
            if document is None:
                raise ValueError("Document not found")
            versions = self._versions(ident)
            parent = versions[-1] if versions else self.legacy(ident, state["runs"])
            runs = {r["id"]: r for r in state["runs"]}
            if any(
                r["document_id"] == ident and r["status"] in {"queued", "restoring", "reviewing"}
                for r in state["runs"]
            ):
                raise ValueError("Wait for this document's current operation to finish")
            if mode == "refine" and body.get("parent_version_id", "legacy") != parent["id"]:
                raise ValueError("Refine the latest document version")
            pages = body.get("pages", list(range(1, len(document["pages"]) + 1)))
            if (
                not isinstance(pages, list)
                or not pages
                or any(type(p) is not int or not 1 <= p <= len(document["pages"]) for p in pages)
                or len(set(pages)) != len(pages)
            ):
                raise ValueError("Select valid document pages")
            if mode == "refine":
                pages = [
                    p
                    for p in pages
                    if (r := self.reviewed_run(runs.get(parent["page_runs"].get(str(p), ""))))
                    and has_repairs(feedback_for(r))
                ]
                if not pages:
                    raise ValueError("Accept a finding or add a comment on a reviewed page first")
            # Validate all model settings before queuing any page requests.
            image_model = body.get("image_model", "codex/gpt-6.1-sol")
            if not isinstance(image_model, str) or not re.fullmatch(
                r"codex/[a-zA-Z0-9._-]+", image_model
            ):
                raise ValueError("Use a Codex model ID for image generation")
            for key in ("generation_prompt", "review_prompt"):
                value = body.get(key)
                if value is not None and (
                    not isinstance(value, str) or not value.strip() or len(value) > 20000
                ):
                    raise ValueError("Prompts must contain 1-20,000 characters")
            if not self.lab.capabilities()["ready"]:
                raise ValueError("Start AgentBridge to process the document")
            version_id = uuid4().hex
            number = parent["number"] + 1
            version = {
                "id": version_id,
                "document_id": ident,
                "number": number,
                "name": f"Version {number}",
                "created_at": datetime.now(UTC).isoformat(),
                "parent_id": parent["id"],
                "page_runs": dict(parent["page_runs"]),
                "changed_pages": pages,
                "mode": mode,
                "max_model_calls": len(pages) * 2,
            }
            directory = self.directory / version_id
            directory.mkdir()
            for page in pages:
                request = {
                    "approved": True,
                    "document_id": ident,
                    "page": page,
                    "name": f"Version {number} · page {page}",
                    "image_model": image_model,
                    "mode": "refine" if mode == "refine" else "restore_and_review",
                }
                for key in ("generation_prompt", "review_prompt"):
                    if key in body:
                        request[key] = body[key]
                if mode == "refine":
                    previous_run = self.reviewed_run(runs.get(parent["page_runs"][str(page)]))
                    assert previous_run is not None
                    request["parent_run_id"] = previous_run["id"]
                run = self.lab.start_run(request, preflighted=True)
                version["page_runs"][str(page)] = run["id"]
            path = directory / "version.json"
            path.write_text(json.dumps(version, indent=2) + "\n")
            return version

    def export(self, document_id: str, version_id: str) -> bytes:
        if not re.fullmatch(r"[a-f0-9]{32}", document_id):
            raise ValueError("Invalid document ID")
        with self.lab.lock:
            state = self.lab.state()
            document = next((d for d in state["documents"] if d["id"] == document_id), None)
            if document is None:
                raise FileNotFoundError
            if version_id == "source":
                page_runs = {}
            elif version_id == "legacy":
                page_runs = self.legacy(document_id, state["runs"])["page_runs"]
            else:
                version = next(
                    (v for v in self._versions(document_id) if v["id"] == version_id), None
                )
                if version is None:
                    raise FileNotFoundError
                page_runs = version["page_runs"]
            runs = {r["id"]: r for r in state["runs"]}
            images = []
            for page in document["pages"]:
                candidate = self.candidate_path(runs.get(page_runs.get(str(page["number"]), "")))
                source = self.lab.root / "documents" / document_id / f"page-{page['number']}.png"
                with Image.open(candidate or source) as image:
                    images.append(image.convert("RGB"))
        # Each download is a separate snapshot; simultaneous exports never overwrite one another.
        export_dir = self.lab.root / "exports"
        export_dir.mkdir(exist_ok=True)
        path = export_dir / f"{uuid4().hex}.pdf"
        original_pdf = self.lab.root / "documents" / document_id / "input.pdf"
        try:
            if original_pdf.is_file():
                build_pdf(original_pdf, path, images)
            else:
                images[0].save(
                    path, format="PDF", save_all=True, append_images=images[1:], resolution=150
                )
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
            for exported_image in images:
                exported_image.close()
