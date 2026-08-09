"""Image decoding, normalization, tiling, and wire-format helpers."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from paperclean.errors import InputError
from paperclean.util import sha256_bytes

WHITE = (255, 255, 255)


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    image: Image.Image
    generated_width: int
    generated_height: int
    effective_dpi: float


def load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot decode image: {path}") from exc
    if image.width < 16 or image.height < 16:
        raise InputError(f"image is too small to clean: {path}")
    return image


def encode_png(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(stream, format="PNG", optimize=False, compress_level=6)
    return stream.getvalue()


def encode_jpeg(image: Image.Image, *, quality: int = 95) -> bytes:
    stream = io.BytesIO()
    image.convert("RGB").save(
        stream,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    return stream.getvalue()


def decode_bytes(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image
    except (OSError, ValueError) as exc:
        raise InputError("the image provider returned an undecodable image") from exc


def encode_for_suffix(image: Image.Image, suffix: str) -> bytes:
    return encode_png(image) if suffix.lower() == ".png" else encode_jpeg(image)


def final_pixel_image(image: Image.Image, suffix: str) -> tuple[bytes, Image.Image]:
    """Encode and decode once so validators inspect the exact published pixels."""
    encoded = encode_for_suffix(image, suffix)
    return encoded, decode_bytes(encoded)


def data_url(image: Image.Image, *, max_edge: int | None = None) -> str:
    wire = image.convert("RGB")
    if max_edge is not None and max(wire.size) > max_edge:
        wire.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return "data:image/png;base64," + base64.b64encode(encode_png(wire)).decode("ascii")


def decode_data_url(value: str) -> bytes:
    if not value.startswith("data:image/") or "," not in value:
        raise InputError("the image provider returned an unsupported image URL")
    header, encoded = value.split(",", 1)
    if ";base64" not in header:
        raise InputError("the image provider returned a non-base64 image")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise InputError("the image provider returned invalid base64") from exc


def normalize_generated(
    generated: Image.Image,
    canvas_size: tuple[int, int],
    *,
    source_dpi: float,
) -> NormalizedImage:
    """Contain generated pixels on the source canvas without cropping or stretching."""
    target_width, target_height = canvas_size
    generated = generated.convert("RGB")
    original_width, original_height = generated.size
    scale = min(target_width / original_width, target_height / original_height)
    width = max(1, round(original_width * scale))
    height = max(1, round(original_height * scale))
    resized = generated.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", canvas_size, WHITE)
    canvas.paste(resized, ((target_width - width) // 2, (target_height - height) // 2))
    effective_dpi = source_dpi * min(original_width / target_width, original_height / target_height)
    return NormalizedImage(
        image=canvas,
        generated_width=original_width,
        generated_height=original_height,
        effective_dpi=effective_dpi,
    )


def review_views(image: Image.Image, *, overlap: float = 0.10) -> list[Image.Image]:
    """Return the full page followed by four overlapping two-by-two tiles."""
    width, height = image.size
    overlap_x = round(width * overlap / 2)
    overlap_y = round(height * overlap / 2)
    midpoint_x = width // 2
    midpoint_y = height // 2
    boxes = [
        (0, 0, min(width, midpoint_x + overlap_x), min(height, midpoint_y + overlap_y)),
        (max(0, midpoint_x - overlap_x), 0, width, min(height, midpoint_y + overlap_y)),
        (0, max(0, midpoint_y - overlap_y), min(width, midpoint_x + overlap_x), height),
        (max(0, midpoint_x - overlap_x), max(0, midpoint_y - overlap_y), width, height),
    ]
    return [image, *(image.crop(box) for box in boxes)]


def review_view_pairs(
    source: Image.Image, candidate: Image.Image
) -> list[tuple[Image.Image, Image.Image]]:
    return list(zip(review_views(source), review_views(candidate), strict=True))


def pixel_sha256(image: Image.Image) -> str:
    header = f"RGB:{image.width}x{image.height}:".encode()
    return sha256_bytes(header + image.convert("RGB").tobytes())


def source_dpi(image: Image.Image, *, default: float = 300.0) -> float:
    raw = image.info.get("dpi")
    if isinstance(raw, tuple) and raw:
        value = float(raw[0])
        if math.isfinite(value) and 36 <= value <= 2400:
            return value
    return default
