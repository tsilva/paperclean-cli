from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest
from pikepdf.canvas import Canvas, Helvetica, Text
from PIL import Image, ImageDraw

from paperclean.errors import UnsafePdfError
from paperclean.pdfs import build_pdf, inspect_pdf, render_overlay_preview, render_pages
from paperclean.provenance import manifest_wrapper


def _blank_pdf(path: Path, *, pages: int = 1) -> None:
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(200, 300))
    pdf.save(path)


def _text_pdf(path: Path) -> None:
    canvas = Canvas(page_size=(200, 300))
    canvas.add_font(pikepdf.Name.F1, Helvetica())
    text = Text()
    text.font(pikepdf.Name.F1, 14)
    text.move_cursor(20, 240)
    text.show("Searchable invoice PC-123")
    canvas.do.draw_text(text)
    pdf = canvas.to_pdf()
    try:
        pdf.save(path)
    finally:
        pdf.close()


def _encrypt_pdf(path: Path, *, user_password: str) -> None:
    with pikepdf.Pdf.open(path, allow_overwriting_input=True) as pdf:
        pdf.save(
            path,
            encryption=pikepdf.Encryption(
                owner="owner-secret",
                user=user_password,
                R=4,
            ),
        )


def test_owner_restricted_pdf_with_empty_user_password_is_supported(tmp_path: Path) -> None:
    source = tmp_path / "owner-restricted.pdf"
    _blank_pdf(source)
    _encrypt_pdf(source, user_password="")

    assert inspect_pdf(source).page_count == 1
    assert len(render_pages(source, dpi=72)) == 1


def test_pdf_with_real_user_password_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "password-protected.pdf"
    _blank_pdf(source)
    _encrypt_pdf(source, user_password="required-secret")

    with pytest.raises(UnsafePdfError, match="encrypted PDFs"):
        inspect_pdf(source)


def test_build_pdf_overlays_all_pages_and_embeds_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _blank_pdf(source, pages=2)
    originals = render_pages(source, dpi=150)
    images = [Image.new("RGB", page.image.size, (250, 250, 250)) for page in originals]
    wrapper = manifest_wrapper({"run_id": "run-1", "schema_version": 1})
    build_pdf(source, output, images, manifest=wrapper, run_id="run-1")
    assert inspect_pdf(output).page_count == 2
    with pikepdf.Pdf.open(output) as pdf:
        assert pdf.attachments["paperclean-run-1.json"].get_file().read_bytes() == __import__(
            "paperclean.provenance", fromlist=["canonical_json"]
        ).canonical_json(wrapper)


def test_rejects_unapplied_redaction(tmp_path: Path) -> None:
    source = tmp_path / "redact.pdf"
    _blank_pdf(source)
    with pikepdf.Pdf.open(source, allow_overwriting_input=True) as pdf:
        annotation = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.Redact,
                Rect=pikepdf.Array([0, 0, 20, 20]),
            )
        )
        pdf.pages[0].Annots = pikepdf.Array([annotation])
        pdf.save(source)
    with pytest.raises(UnsafePdfError, match="redaction"):
        inspect_pdf(source)


def test_rejects_javascript_actions(tmp_path: Path) -> None:
    source = tmp_path / "javascript.pdf"
    _blank_pdf(source)
    with pikepdf.Pdf.open(source, allow_overwriting_input=True) as pdf:
        pdf.Root.OpenAction = pdf.make_indirect(
            pikepdf.Dictionary(S=pikepdf.Name.JavaScript, JS="app.alert('x')")
        )
        pdf.save(source)
    with pytest.raises(UnsafePdfError, match="JavaScript"):
        inspect_pdf(source)


def test_raster_overlay_preserves_searchable_text_layer(tmp_path: Path) -> None:
    source = tmp_path / "source-text.pdf"
    output = tmp_path / "output-text.pdf"
    _text_pdf(source)
    original = render_pages(source, dpi=150)[0]
    assert "Searchable invoice PC-123" in original.text_signature
    build_pdf(source, output, [original.image])
    final = render_pages(output, dpi=150)[0]
    assert final.text_signature == original.text_signature


def test_overlay_preview_contains_only_selected_page(tmp_path: Path) -> None:
    source = tmp_path / "multi-page.pdf"
    preview = tmp_path / "preview.pdf"
    _blank_pdf(source, pages=5)
    original = render_pages(source, dpi=72)[3]
    candidate = Image.new("RGB", original.image.size, "red")

    rendered = render_overlay_preview(source, 3, candidate, preview, dpi=72)

    assert inspect_pdf(preview).page_count == 1
    assert rendered.getpixel((rendered.width // 2, rendered.height // 2)) == (255, 0, 0)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_overlay_respects_rotated_page_geometry(tmp_path: Path, rotation: int) -> None:
    source = tmp_path / f"source-{rotation}.pdf"
    output = tmp_path / f"output-{rotation}.pdf"
    _blank_pdf(source)
    with pikepdf.Pdf.open(source, allow_overwriting_input=True) as pdf:
        pdf.pages[0].obj.Rotate = rotation
        pdf.save(source)
    page = render_pages(source, dpi=72)[0]
    candidate = Image.new("RGB", page.image.size, "white")
    draw = ImageDraw.Draw(candidate)
    width, height = candidate.size
    draw.rectangle((0, 0, width // 2, height // 2), fill="red")
    draw.rectangle((width // 2, 0, width, height // 2), fill="green")
    draw.rectangle((0, height // 2, width // 2, height), fill="blue")
    draw.rectangle((width // 2, height // 2, width, height), fill="yellow")
    build_pdf(source, output, [candidate])
    rendered = render_pages(output, dpi=72)[0].image
    assert rendered.size == candidate.size
    samples = [
        rendered.getpixel((10, 10)),
        rendered.getpixel((width - 10, 10)),
        rendered.getpixel((10, height - 10)),
        rendered.getpixel((width - 10, height - 10)),
    ]
    assert samples == [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0)]
