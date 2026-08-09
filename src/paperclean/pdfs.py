"""Static PDF rendering, safety checks, raster overlays, and provenance."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf
import pypdfium2 as pdfium  # type: ignore[import-untyped]
from pikepdf import Dictionary, Pdf, Rectangle
from pikepdf.canvas import Canvas
from PIL import Image

from paperclean.errors import InputError, UnsafePdfError
from paperclean.provenance import canonical_json

MAX_RENDER_EDGE = 6000


@dataclass(slots=True)
class PdfInspection:
    page_count: int
    removed_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RenderedPage:
    image: Image.Image
    dpi: float
    rotation: int
    cropbox: tuple[float, float, float, float]
    user_unit: float
    text_signature: str


def _get(dictionary: Any, key: str) -> Any | None:
    try:
        return dictionary.get(key)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _name(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # pikepdf can throw on malformed strings
        return ""


def _inherited(page: pikepdf.Page, key: str, default: Any) -> Any:
    node: Any = page.obj
    seen: set[tuple[int, int]] = set()
    while node is not None:
        value = _get(node, key)
        if value is not None:
            return value
        objgen = getattr(node, "objgen", None)
        if objgen in seen:
            break
        if objgen is not None:
            seen.add(objgen)
        node = _get(node, "/Parent")
    return default


def _box(page: pikepdf.Page) -> tuple[float, float, float, float]:
    raw = _inherited(page, "/CropBox", _inherited(page, "/MediaBox", [0, 0, 612, 792]))
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise UnsafePdfError("PDF page has an invalid page box") from exc
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        raise UnsafePdfError("PDF page has an invalid page box")
    return values


def _rotation(page: pikepdf.Page) -> int:
    try:
        value = int(_inherited(page, "/Rotate", 0)) % 360
    except (TypeError, ValueError):
        value = 0
    if value not in {0, 90, 180, 270}:
        raise UnsafePdfError("PDF page rotation is not a multiple of 90 degrees")
    return value


def _user_unit(page: pikepdf.Page) -> float:
    try:
        value = float(_inherited(page, "/UserUnit", 1.0))
    except (TypeError, ValueError) as exc:
        raise UnsafePdfError("PDF page has an invalid UserUnit") from exc
    if not 0.01 <= value <= 75_000:
        raise UnsafePdfError("PDF page has an unsafe UserUnit")
    return value


def inspect_pdf(path: Path) -> PdfInspection:
    try:
        pdf = Pdf.open(path, inherit_page_attributes=False)
    except pikepdf.PasswordError as exc:
        raise UnsafePdfError("encrypted PDFs are not supported") from exc
    except pikepdf.PdfError as exc:
        raise InputError(f"cannot open PDF: {path}") from exc
    with pdf:
        if pdf.is_encrypted:
            raise UnsafePdfError("encrypted PDFs are not supported")
        if not pdf.pages:
            raise InputError("PDF has no pages")
        removed: set[str] = set()
        warnings: list[str] = []
        root = pdf.Root
        acroform = _get(root, "/AcroForm")
        if acroform is not None:
            if _get(acroform, "/XFA") is not None:
                raise UnsafePdfError("XFA forms are not supported")
            if _get(acroform, "/CO") is not None:
                raise UnsafePdfError("calculated PDF forms are not supported")
            removed.add("interactive_forms")
        if _get(root, "/OpenAction") is not None or _get(root, "/AA") is not None:
            removed.add("document_actions")
        names = _get(root, "/Names")
        if names is not None:
            if _get(names, "/JavaScript") is not None:
                raise UnsafePdfError("PDF JavaScript is not supported")
            if _get(names, "/EmbeddedFiles") is not None:
                removed.add("embedded_files")
        if _get(root, "/AF") is not None:
            removed.add("associated_files")
        for obj in pdf.objects:
            if not isinstance(obj, Dictionary):
                continue
            if _name(_get(obj, "/S")) == "/JavaScript" or _get(obj, "/JS") is not None:
                raise UnsafePdfError("PDF JavaScript is not supported")
        for page in pdf.pages:
            _box(page)
            _rotation(page)
            _user_unit(page)
            if _get(page.obj, "/AA") is not None:
                removed.add("page_actions")
            annots = _get(page.obj, "/Annots")
            if annots is None:
                continue
            removed.add("annotations")
            for annotation in annots:
                subtype = _name(_get(annotation, "/Subtype"))
                if subtype == "/Redact":
                    raise UnsafePdfError("PDF contains unapplied redaction annotations")
                if subtype == "/RichMedia":
                    removed.add("rich_media")
                if subtype == "/Widget":
                    appearance = _get(annotation, "/AP")
                    if appearance is None or _get(appearance, "/N") is None:
                        raise UnsafePdfError("PDF widget has no static appearance stream")
                    if _name(_get(annotation, "/FT")) == "/Sig":
                        removed.add("digital_signatures")
        metadata = _get(root, "/Metadata")
        if metadata is not None:
            try:
                raw = bytes(metadata.read_bytes())
            except (AttributeError, pikepdf.PdfError):
                raw = b""
            if re.search(rb"pdfaid:|pdfxid:|pdfx:", raw, flags=re.IGNORECASE):
                removed.add("invalidated_archival_conformance_claims")
                warnings.append(
                    "PDF/A or PDF/X claims were removed because raster overlays invalidate them"
                )
        return PdfInspection(len(pdf.pages), sorted(removed), warnings)


def page_count(path: Path) -> int:
    return inspect_pdf(path).page_count


def _text_signature(page: pdfium.PdfPage) -> str:
    text_page = page.get_textpage()
    try:
        text = text_page.get_text_range()
    finally:
        text_page.close()
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n")


def render_pages(path: Path, *, dpi: int) -> list[RenderedPage]:
    inspection = inspect_pdf(path)
    del inspection
    with Pdf.open(path, inherit_page_attributes=False) as structure:
        geometries = [(_box(page), _rotation(page), _user_unit(page)) for page in structure.pages]
    document = pdfium.PdfDocument(path)
    try:
        document.init_forms()
        rendered: list[RenderedPage] = []
        for index in range(len(document)):
            page = document[index]
            try:
                box, rotation, user_unit = geometries[index]
                scale = dpi * user_unit / 72
                width_points, height_points = page.get_size()
                projected = max(width_points, height_points) * scale
                if projected > MAX_RENDER_EDGE:
                    scale *= MAX_RENDER_EDGE / projected
                bitmap = page.render(scale=scale, may_draw_forms=True, draw_annots=True)
                try:
                    image = bitmap.to_pil().convert("RGB")
                    image.load()
                finally:
                    bitmap.close()
                effective_dpi = scale * 72 / user_unit
                rendered.append(
                    RenderedPage(
                        image=image,
                        dpi=effective_dpi,
                        rotation=rotation,
                        cropbox=box,
                        user_unit=user_unit,
                        text_signature=_text_signature(page),
                    )
                )
            finally:
                page.close()
        return rendered
    finally:
        document.close()


def _overlay_page(page: pikepdf.Page, image: Image.Image) -> None:
    cropbox = _box(page)
    rotation = _rotation(page)
    width = cropbox[2] - cropbox[0]
    height = cropbox[3] - cropbox[1]
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    canvas = Canvas(page_size=(display_width, display_height))
    canvas.do.draw_image(image.convert("RGB"), 0, 0, display_width, display_height)
    overlay = canvas.to_pdf()
    try:
        page.add_overlay(  # type: ignore[call-arg]
            overlay.pages[0],
            rect=Rectangle(*cropbox),
            push_stack=True,
            shrink=False,
            expand=False,
        )
    finally:
        overlay.close()


def _delete_key(dictionary: Any, key: str) -> bool:
    try:
        if key in dictionary:
            del dictionary[key]
            return True
    except (KeyError, TypeError, ValueError):
        pass
    return False


def _sanitize_outline(node: Any, seen: set[tuple[int, int]]) -> None:
    while node is not None:
        objgen = getattr(node, "objgen", None)
        if objgen is not None and objgen in seen:
            return
        if objgen is not None:
            seen.add(objgen)
        _delete_key(node, "/A")
        child = _get(node, "/First")
        if child is not None:
            _sanitize_outline(child, seen)
        node = _get(node, "/Next")


def _sanitize_pdf(pdf: Pdf) -> None:
    root = pdf.Root
    for key in ("/AcroForm", "/OpenAction", "/AA", "/AF"):
        _delete_key(root, key)
    names = _get(root, "/Names")
    if names is not None:
        for key in ("/JavaScript", "/EmbeddedFiles"):
            _delete_key(names, key)
        if not list(names.keys()):
            _delete_key(root, "/Names")
    outlines = _get(root, "/Outlines")
    if outlines is not None:
        first = _get(outlines, "/First")
        if first is not None:
            _sanitize_outline(first, set())
    for page in pdf.pages:
        _delete_key(page.obj, "/Annots")
        _delete_key(page.obj, "/AA")
    try:
        with pdf.open_metadata(set_pikepdf_as_editor=False) as metadata:
            for key in list(metadata):
                lowered = key.lower()
                if "pdfaid" in lowered or "pdfx" in lowered:
                    del metadata[key]
    except (ValueError, pikepdf.PdfError):
        pass


def build_pdf(
    source: Path,
    destination: Path,
    page_images: Iterable[Image.Image],
    *,
    manifest: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    try:
        pdf = Pdf.open(source, inherit_page_attributes=False)
    except pikepdf.PdfError as exc:
        raise InputError(f"cannot reopen PDF: {source}") from exc
    with pdf:
        images = list(page_images)
        if len(images) != len(pdf.pages):
            raise ValueError("PDF output page count does not match source")
        for page, image in zip(pdf.pages, images, strict=True):
            _overlay_page(page, image)
        _sanitize_pdf(pdf)
        if manifest is not None:
            if not run_id:
                raise ValueError("run_id is required with a PDF manifest")
            pdf.attachments[f"paperclean-{run_id}.json"] = canonical_json(manifest)
        pdf.save(
            destination,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
            linearize=False,
        )


def render_overlay_preview(
    source: Path,
    page_index: int,
    candidate: Image.Image,
    destination: Path,
    *,
    dpi: int,
) -> Image.Image:
    """Render a scratch copy so review sees the actual PDF composition path."""
    originals = render_pages(source, dpi=dpi)
    images = [page.image for page in originals]
    images[page_index] = candidate
    build_pdf(source, destination, images)
    return render_pages(destination, dpi=dpi)[page_index].image
