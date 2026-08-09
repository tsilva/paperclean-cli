from __future__ import annotations

import io
import struct

from PIL import Image

from paperclean.imaging import normalize_generated, review_views
from paperclean.provenance import (
    embed_jpeg,
    embed_png,
    extract_jpeg,
    extract_png,
    manifest_wrapper,
)


def _image_bytes(image_format: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (80, 120), (245, 244, 240)).save(output, format=image_format)
    return output.getvalue()


def _png_idat(data: bytes) -> list[bytes]:
    chunks: list[bytes] = []
    offset = 8
    while offset + 12 <= len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        if kind == b"IDAT":
            chunks.append(data[offset + 8 : offset + 8 + size])
        offset += size + 12
    return chunks


def test_jpeg_manifest_round_trip_preserves_scan_data() -> None:
    original = _image_bytes("JPEG")
    wrapper = manifest_wrapper({"run_id": "abc", "schema_version": 1})
    embedded = embed_jpeg(original, wrapper)
    assert extract_jpeg(embedded) == wrapper
    assert embedded.endswith(original[2:])


def test_png_manifest_round_trip_preserves_idat() -> None:
    original = _image_bytes("PNG")
    wrapper = manifest_wrapper({"run_id": "abc", "schema_version": 1})
    embedded = embed_png(original, wrapper)
    assert extract_png(embedded) == wrapper
    assert _png_idat(embedded) == _png_idat(original)


def test_normalization_contains_and_pads_without_stretching() -> None:
    generated = Image.new("RGB", (200, 100), "black")
    normalized = normalize_generated(generated, (100, 100), source_dpi=300)
    assert normalized.image.size == (100, 100)
    assert normalized.image.getpixel((50, 10)) == (255, 255, 255)
    assert normalized.image.getpixel((50, 50)) == (0, 0, 0)
    assert normalized.effective_dpi == 300


def test_review_views_are_full_page_plus_four_regions() -> None:
    views = review_views(Image.new("RGB", (1000, 800), "white"))
    assert len(views) == 5
    assert views[0].size == (1000, 800)
    assert all(view.width > 500 and view.height > 400 for view in views[1:])
