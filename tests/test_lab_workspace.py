from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any

import pikepdf
import pytest
from pikepdf.canvas import Canvas, Helvetica, Text
from PIL import Image

from paperclean.imaging import decode_data_url
from paperclean.lab import Lab
from paperclean.pdfs import render_pages


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (200, 300), color).save(output, format="PNG")
    return output.getvalue()


def document_pdf(path: Path) -> bytes:
    pdf = pikepdf.Pdf.new()
    for number in (1, 2):
        canvas = Canvas(page_size=(200, 300))
        canvas.add_font(pikepdf.Name.F1, Helvetica())
        text = (
            Text().font(pikepdf.Name.F1, 12).move_cursor(15, 250).show(f"Searchable page {number}")
        )
        canvas.do.draw_text(text)
        with canvas.to_pdf() as page:
            pdf.pages.append(page.pages[0])
    pdf.save(path)
    pdf.close()
    return path.read_bytes()


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Lab, dict[str, Any], list[Any]]:
    lab = Lab(tmp_path / "data", "http://127.0.0.1:8082/api/v1")
    monkeypatch.setattr(lab, "capabilities", lambda: {"ready": True})
    calls: list[Any] = []

    def request(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls.append((path, payload))
        if path == "/images":
            color = "green" if "User feedback:" in payload["prompt"] else "blue"
            return {"data": [{"b64_json": base64.b64encode(image_bytes(color)).decode()}]}
        issue = {
            "label": "Fixture finding",
            "category": "changed_content",
            "severity": "low",
            "description": "A test finding",
            "source_evidence": "Fixture evidence",
            "restored_box": {"x0": 0.1, "y0": 0.1, "x1": 0.2, "y1": 0.2},
            "source_box": None,
        }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"summary": "Fixture review", "issues": [issue]})
                    }
                }
            ]
        }

    monkeypatch.setattr(lab, "request", request)
    doc = lab.import_document("document.pdf", document_pdf(tmp_path / "source.pdf"))
    return lab, doc, calls


def wait(lab: Lab) -> None:
    lab.executor.submit(lambda: None).result(timeout=10)


def test_refinement_uses_latest_pixels_and_frozen_user_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab, doc, calls = setup(tmp_path, monkeypatch)
    try:
        first = lab.workspace.create({"approved": True, "document_id": doc["id"]})
        wait(lab)
        page_one = first["page_runs"]["1"]
        lab.judgments(
            page_one,
            {
                "judgments": {"1": "confirmed"},
                "comments": {"1": "Repair this region"},
                "notes": "Keep everything else",
            },
        )
        next_version = lab.workspace.create(
            {
                "approved": True,
                "document_id": doc["id"],
                "mode": "refine",
                "parent_version_id": first["id"],
            }
        )
        wait(lab)
        assert next_version["changed_pages"] == [1]
        assert next_version["page_runs"]["2"] == first["page_runs"]["2"]
        new_run = next(r for r in lab.state()["runs"] if r["id"] == next_version["page_runs"]["1"])
        assert new_run["parent_run_id"] == page_one
        assert new_run["refinement_feedback"]["accepted_issues"][0]["id"] == 1
        assert new_run["comments"] == {}
        image_request = [body for path, body in calls if path == "/images"][-1]
        assert len(image_request["input_references"]) == 1
        previous = Image.open(
            io.BytesIO(decode_data_url(image_request["input_references"][0]["image_url"]["url"]))
        )
        assert previous.getpixel((doc["pages"][0]["width"] + 29, 53)) == (0, 0, 255)
        assert "Repair this region" in image_request["prompt"]
        # Later annotation edits do not change feedback already applied to a version.
        lab.judgments(
            page_one, {"judgments": {"1": "false_alarm"}, "comments": {"1": "Different comment"}}
        )
        assert (
            new_run["refinement_feedback"]["issue_comments"][0]["comment"] == "Repair this region"
        )
        lab.judgments(new_run["id"], {"judgments": {}, "notes": "Another refinement"})
        third = lab.workspace.create(
            {
                "approved": True,
                "document_id": doc["id"],
                "mode": "refine",
                "parent_version_id": next_version["id"],
            }
        )
        wait(lab)
        assert third["number"] == 3
        assert third["page_runs"]["2"] == first["page_runs"]["2"]
        latest_request = [body for path, body in calls if path == "/images"][-1]
        prior = Image.open(
            io.BytesIO(decode_data_url(latest_request["input_references"][0]["image_url"]["url"]))
        )
        assert prior.getpixel((doc["pages"][0]["width"] + 29, 53)) == (0, 128, 0)
        with pytest.raises(ValueError, match="latest document version"):
            lab.workspace.create(
                {
                    "approved": True,
                    "document_id": doc["id"],
                    "mode": "refine",
                    "parent_version_id": first["id"],
                }
            )
    finally:
        lab.executor.shutdown(wait=True)


def test_export_during_repair_uses_previous_page_and_preserves_all_pages_and_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab, doc, _ = setup(tmp_path, monkeypatch)
    gate, started = threading.Event(), threading.Event()
    try:
        first = lab.workspace.create({"approved": True, "document_id": doc["id"], "pages": [1]})
        wait(lab)
        lab.judgments(first["page_runs"]["1"], {"judgments": {}, "notes": "Repair page"})
        real_request = lab.request

        def blocked(path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if path == "/images":
                started.set()
                assert gate.wait(10)
            return real_request(path, payload)

        monkeypatch.setattr(lab, "request", blocked)
        second = lab.workspace.create(
            {
                "approved": True,
                "document_id": doc["id"],
                "mode": "refine",
                "parent_version_id": first["id"],
            }
        )
        assert started.wait(5)
        exported = tmp_path / "export.pdf"
        exported.write_bytes(lab.workspace.export(doc["id"], second["id"]))
        pages = render_pages(exported, dpi=72)
        assert len(pages) == 2
        assert "Searchable page 1" in pages[0].text_signature
        assert "Searchable page 2" in pages[1].text_signature
        assert pages[0].image.getpixel((5, 5)) == (0, 0, 255)
        assert pages[1].image.getpixel((5, 5)) == (255, 255, 255)
        assert not list((lab.root / "exports").glob("*.pdf"))
        gate.set()
        wait(lab)
        newer = tmp_path / "newer.pdf"
        newer.write_bytes(lab.workspace.export(doc["id"], second["id"]))
        assert render_pages(newer, dpi=72)[0].image.getpixel((5, 5)) == (0, 128, 0)
        # Exporting old versions is reproducible after further refinements.
        old_export = tmp_path / "old-version.pdf"
        old_export.write_bytes(lab.workspace.export(doc["id"], first["id"]))
        assert render_pages(old_export, dpi=72)[0].image.getpixel((5, 5)) == (0, 0, 255)
    finally:
        gate.set()
        lab.executor.shutdown(wait=True)


def test_repair_rejects_cross_document_parent_and_requires_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab, doc, _ = setup(tmp_path, monkeypatch)
    try:
        first = lab.workspace.create({"approved": True, "document_id": doc["id"]})
        wait(lab)
        with pytest.raises(ValueError, match="Accept a finding"):
            lab.workspace.create(
                {
                    "approved": True,
                    "document_id": doc["id"],
                    "mode": "refine",
                    "parent_version_id": first["id"],
                }
            )
        other = lab.import_document("other.png", image_bytes("white"))
        with pytest.raises(ValueError, match="belong to this document and page"):
            lab.start_run(
                {
                    "approved": True,
                    "document_id": other["id"],
                    "page": 1,
                    "mode": "refine",
                    "parent_run_id": first["page_runs"]["1"],
                }
            )
        with pytest.raises(ValueError, match="Invalid issue comments"):
            lab.judgments(first["page_runs"]["1"], {"comments": {"999": "Invalid region"}})
    finally:
        lab.executor.shutdown(wait=True)


def test_legacy_history_is_frozen_and_failed_repairs_can_be_refined_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab, doc, _ = setup(tmp_path, monkeypatch)
    try:
        old = lab.start_run({"approved": True, "document_id": doc["id"], "page": 1})
        wait(lab)
        lab.judgments(old["id"], {"judgments": {}, "notes": "Preserve supported content"})
        real_request = lab.request

        def failing(path: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("Fixture image generation failure")

        monkeypatch.setattr(lab, "request", failing)
        failed = lab.workspace.create(
            {
                "approved": True,
                "document_id": doc["id"],
                "mode": "refine",
                "parent_version_id": "legacy",
            }
        )
        wait(lab)
        assert (
            next(v for v in lab.state()["versions"] if v["id"] == failed["id"])["status"]
            == "needs_attention"
        )
        monkeypatch.setattr(lab, "request", real_request)
        repaired = lab.workspace.create(
            {
                "approved": True,
                "document_id": doc["id"],
                "mode": "refine",
                "parent_version_id": failed["id"],
            }
        )
        wait(lab)
        history = next(v for v in lab.state()["versions"] if v["id"] == "legacy")
        assert history["page_runs"]["1"] == old["id"]
        assert history["status"] == "partial"
        latest = next(r for r in lab.state()["runs"] if r["id"] == repaired["page_runs"]["1"])
        assert latest["parent_run_id"] == old["id"]
        old_pdf = tmp_path / "legacy.pdf"
        old_pdf.write_bytes(lab.workspace.export(doc["id"], "legacy"))
        assert render_pages(old_pdf, dpi=72)[0].image.getpixel((5, 5)) == (0, 0, 255)
    finally:
        lab.executor.shutdown(wait=True)
