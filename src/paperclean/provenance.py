"""Compact embedded provenance and verbose sidecar reports."""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

JPEG_IDENTIFIER = b"PaperClean\x00"
PNG_KEYWORD = b"paperclean.manifest.v1"
MAX_EMBEDDED_MANIFEST = 48 * 1024


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON for the integer/string-only manifest profile."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_wrapper(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = canonical_json(payload)
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def encoded_manifest(wrapper: dict[str, Any]) -> bytes:
    encoded = canonical_json(wrapper)
    if len(encoded) > MAX_EMBEDDED_MANIFEST:
        raise ValueError("embedded manifest exceeds the 48 KiB limit")
    return encoded


def embed_jpeg(data: bytes, wrapper: dict[str, Any]) -> bytes:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG stream")
    body = JPEG_IDENTIFIER + encoded_manifest(wrapper)
    if len(body) + 2 > 0xFFFF:
        raise ValueError("JPEG APP15 PaperClean manifest is too large")
    segment = b"\xff\xef" + struct.pack(">H", len(body) + 2) + body
    return data[:2] + segment + data[2:]


def extract_jpeg(data: bytes) -> dict[str, Any] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 4 <= len(data) and data[offset] == 0xFF:
        marker = data[offset + 1]
        if marker in {0xD9, 0xDA}:
            break
        length = struct.unpack(">H", data[offset + 2 : offset + 4])[0]
        end = offset + 2 + length
        if end > len(data) or length < 2:
            return None
        body = data[offset + 4 : end]
        if marker == 0xEF and body.startswith(JPEG_IDENTIFIER):
            value = json.loads(body[len(JPEG_IDENTIFIER) :])
            return value if isinstance(value, dict) else None
        offset = end
    return None


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def embed_png(data: bytes, wrapper: dict[str, Any]) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError("not a PNG stream")
    text = encoded_manifest(wrapper)
    payload = PNG_KEYWORD + b"\x00\x00\x00\x00\x00" + text
    chunk = _png_chunk(b"iTXt", payload)
    offset = len(signature)
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        if kind == b"IEND":
            return data[:offset] + chunk + data[offset:]
        offset += 12 + length
    raise ValueError("PNG has no IEND chunk")


def extract_png(data: bytes) -> dict[str, Any] | None:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        return None
    offset = len(signature)
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        if kind == b"iTXt" and payload.startswith(PNG_KEYWORD + b"\x00"):
            parts = payload.split(b"\x00", 5)
            if len(parts) == 6:
                value = json.loads(parts[5])
                return value if isinstance(value, dict) else None
        offset += 12 + length
    return None


def embed_image(data: bytes, suffix: str, wrapper: dict[str, Any]) -> bytes:
    if suffix.lower() in {".jpg", ".jpeg"}:
        return embed_jpeg(data, wrapper)
    if suffix.lower() == ".png":
        return embed_png(data, wrapper)
    raise ValueError(f"unsupported provenance image type: {suffix}")


def write_report(path: Path, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    from paperclean.util import private_write

    private_write(path, encoded + b"\n")
