"""Pristine reconstruction helpers that preserve hard-to-read source content."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageFilter

from paperclean.imaging import (
    finish_pristine_recreation,
    normalize_generated,
    review_view_pairs,
)
from paperclean.models import Discrepancy
from paperclean.prompting import REGIONAL_REPAIR_PROMPT
from paperclean.validation import OcrToken, _registration_matrix, ocr_tokens

REPAIRABLE_CATEGORIES = {
    "changed_text",
    "missing_text",
    "invented_text",
    "changed_handwriting",
    "changed_signature",
    "changed_stamp",
    "changed_redaction",
    "changed_table",
    "changed_diagram",
    "unresolved_content",
}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class ImageGenerator(Protocol):
    def generate(self, source: Image.Image, prompt: str, *, max_edge: int) -> Image.Image: ...


def _pixel_matrix(normalized: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.array([[width, 0, 0], [0, height, 0], [0, 0, 1]], dtype=np.float64)
    return np.asarray(scale @ normalized @ np.linalg.inv(scale), dtype=np.float64)


def clear_page_border(image: Image.Image, *, fraction: float = 0.008) -> Image.Image:
    """Remove thin model-generated paper/capture remnants at the canvas boundary."""
    result = np.asarray(image.convert("RGB")).copy()
    margin_x = max(1, round(image.width * fraction))
    margin_y = max(1, round(image.height * fraction))
    result[:margin_y, :] = 255
    result[-margin_y:, :] = 255
    result[:, :margin_x] = 255
    result[:, -margin_x:] = 255
    return Image.fromarray(result, "RGB")


def registered_review_pairs(
    source: Image.Image, candidate: Image.Image
) -> list[tuple[Image.Image, Image.Image]]:
    """Align regional source views to a de-skewed candidate before cropping."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    pairs = review_view_pairs(source, candidate)
    if source.size != candidate.size:
        return pairs
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    candidate_to_source = _registration_matrix(source_gray, candidate_gray)
    if candidate_to_source is None:
        return pairs
    try:
        source_to_candidate = np.linalg.inv(candidate_to_source)
    except np.linalg.LinAlgError:
        return pairs
    matrix = _pixel_matrix(source_to_candidate, source.width, source.height)
    registered = cv2.warpPerspective(
        np.asarray(source),
        matrix,
        candidate.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    regional = review_view_pairs(Image.fromarray(registered, "RGB"), candidate)
    # The full-page view remains authoritative and shows the reviewer the exact input.
    return [pairs[0], *regional[1:]]


def _clean_source(source: Image.Image) -> np.ndarray:
    pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    sigma = max(source.size) / 45
    background = cv2.GaussianBlur(pixels, (0, 0), sigmaX=sigma, sigmaY=sigma)
    cleaned = np.clip(pixels * 252 / np.maximum(background, 32), 0, 255).astype(np.uint8)
    # Recover edge contrast without synthesizing replacement glyphs. This preserves
    # every source pixel's shape while making faint microprint look freshly printed.
    cleaned = (255 - np.clip((255 - cleaned.astype(np.int16)) * 2.4, 0, 255)).astype(np.uint8)
    cleaned = np.asarray(
        Image.fromarray(cleaned, "RGB").filter(
            ImageFilter.UnsharpMask(radius=0.65, percent=180, threshold=1)
        )
    ).copy()
    gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)
    chroma = cleaned.max(axis=2).astype(np.int16) - cleaned.min(axis=2).astype(np.int16)
    cleaned[(gray >= 238) & (chroma <= 10)] = 255
    return cleaned


def _remove_paper_tone(pixels: np.ndarray) -> np.ndarray:
    """Whiten scan paper after interpolation without touching dark authored ink."""
    result = pixels.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    red = result[:, :, 0].astype(np.int16)
    green = result[:, :, 1].astype(np.int16)
    blue = result[:, :, 2].astype(np.int16)
    yellow_paper = (red > blue + 8) & (green > blue + 4) & (gray >= 180)
    result[(gray >= 225) | yellow_paper] = 255
    return result


def _remove_scanner_borders(pixels: np.ndarray) -> np.ndarray:
    """Remove boundary-connected scan rails while retaining nearby small ink."""
    result = pixels.copy()
    height, width = result.shape[:2]
    edge = max(1, round(width * 0.012))
    result[:, :edge] = 255
    result[:, -edge:] = 255
    dark = (cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) < 140).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for label in range(1, count):
        x, _y, component_width, component_height, _area = stats[label]
        near_side = x < width * 0.04 or x + component_width > width * 0.96
        border_bar = component_height > height * 0.04 and component_height > component_width * 4
        if near_side and border_bar:
            component = (labels == label).astype(np.uint8)
            component = cv2.dilate(component, np.ones((5, 5), np.uint8))
            result[component > 0] = 255
    return result


def _nearest_edge(box: tuple[float, float, float, float]) -> str:
    left, top, right, bottom = box
    distances = {"left": left, "top": top, "right": 1 - right, "bottom": 1 - bottom}
    return min(distances, key=distances.__getitem__)


def _edge_distance(box: tuple[float, float, float, float], edge: str) -> float:
    left, top, right, bottom = box
    return {"left": left, "top": top, "right": 1 - right, "bottom": 1 - bottom}[edge]


def _token_region(
    tokens: Sequence[OcrToken],
    *,
    horizontal_padding: float,
    vertical_padding: float,
) -> tuple[float, float, float, float]:
    widths = [token.box[2] - token.box[0] for token in tokens]
    heights = [token.box[3] - token.box[1] for token in tokens]
    pad_x = float(np.median(widths)) * horizontal_padding
    pad_y = float(np.median(heights)) * vertical_padding
    return (
        max(0.0, min(token.box[0] for token in tokens) - pad_x),
        max(0.0, min(token.box[1] for token in tokens) - pad_y),
        min(1.0, max(token.box[2] for token in tokens) + pad_x),
        min(1.0, max(token.box[3] for token in tokens) + pad_y),
    )


def _outer_microprint(tokens: Sequence[OcrToken], edge: str) -> list[OcrToken]:
    """Find an outermost small-text cluster from the page's own OCR distribution."""
    reliable_heights = [token.box[3] - token.box[1] for token in tokens if token.confidence >= 60]
    if not reliable_heights:
        return []
    typical_height = float(np.median(reliable_heights))
    edge_tokens = [token for token in tokens if _nearest_edge(token.box) == edge]
    if not edge_tokens:
        return []
    ordered = sorted(edge_tokens, key=lambda token: _edge_distance(token.box, edge))
    gaps = [
        _edge_distance(right.box, edge) - _edge_distance(left.box, edge)
        for left, right in pairwise(ordered)
    ]
    meaningful = [(gap, index) for index, gap in enumerate(gaps) if gap > typical_height * 3]
    cluster = ordered[: max(meaningful)[1] + 1] if meaningful else ordered
    cluster_height = float(np.median([token.box[3] - token.box[1] for token in cluster]))
    if (
        _edge_distance(cluster[0].box, edge) > typical_height * 8
        or cluster_height > typical_height * 1.1
    ):
        return []
    return cluster


def _paint_region(
    mask: np.ndarray, region: tuple[float, float, float, float], value: int = 255
) -> None:
    height, width = mask.shape
    left, top, right, bottom = region
    cv2.rectangle(
        mask,
        (max(0, round(left * width)), max(0, round(top * height))),
        (min(width - 1, round(right * width)), min(height - 1, round(bottom * height))),
        value,
        thickness=-1,
    )


def _edge_model_is_authoritative(
    expected: Sequence[OcrToken],
    actual: Sequence[OcrToken],
    candidate_to_source: np.ndarray,
    *,
    edge: str,
) -> bool:
    """Require exact, spatially registered OCR before retaining generated edge text."""
    if not expected:
        return False

    expected_region = _token_region(
        expected,
        horizontal_padding=12,
        vertical_padding=12,
    )

    registered: list[tuple[OcrToken, float, float]] = []
    for token in actual:
        center_x = (token.box[0] + token.box[2]) / 2
        center_y = (token.box[1] + token.box[3]) / 2
        point = candidate_to_source @ np.array([center_x, center_y, 1.0], dtype=np.float64)
        if abs(point[2]) < 1e-9:
            continue
        mapped_x = float(point[0] / point[2])
        mapped_y = float(point[1] / point[2])
        in_edge = (
            _nearest_edge((mapped_x, mapped_y, mapped_x, mapped_y)) == edge
            and expected_region[0] <= mapped_x <= expected_region[2]
            and expected_region[1] <= mapped_y <= expected_region[3]
        )
        if in_edge:
            registered.append((token, mapped_x, mapped_y))

    used: set[int] = set()
    for expected_token in expected:
        if expected_token.confidence < 60:
            return False
        expected_x = (expected_token.box[0] + expected_token.box[2]) / 2
        expected_y = (expected_token.box[1] + expected_token.box[3]) / 2
        choices = [
            (
                max(abs(expected_x - mapped_x), abs(expected_y - mapped_y)),
                index,
            )
            for index, (token, mapped_x, mapped_y) in enumerate(registered)
            if index not in used and token.confidence >= 60 and token.text == expected_token.text
        ]
        if not choices:
            return False
        distance, index = min(choices)
        if distance > 0.025:
            return False
        used.add(index)
    # Reject duplicated/invented model tokens as well as missing source tokens.
    return len(used) == len(registered)


def restore_source_regions(
    source: Image.Image,
    candidate: Image.Image,
    regions: Sequence[tuple[float, float, float, float]],
) -> Image.Image:
    """Preserve authored source pixels in selected regions on a clean white ground."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size or not regions:
        return candidate
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    candidate_to_source = _registration_matrix(source_gray, candidate_gray)
    if candidate_to_source is None:
        return candidate
    try:
        source_to_candidate = np.linalg.inv(candidate_to_source)
    except np.linalg.LinAlgError:
        return candidate
    matrix = _pixel_matrix(source_to_candidate, source.width, source.height)
    registered = cv2.warpPerspective(
        _clean_source(source),
        matrix,
        candidate.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    # Scanner paper and its yellow/brown noise are not content. A firm white cutoff
    # leaves dark glyphs and pen strokes crisp instead of carrying a dirty rectangle.
    registered = _remove_paper_tone(registered)
    registered = _remove_scanner_borders(registered)

    result = np.asarray(candidate).copy()
    width, height = candidate.size
    for left, top, right, bottom in regions:
        pad_x = max(0.004, (right - left) * 0.08)
        pad_y = max(0.004, (bottom - top) * 0.12)
        x1 = max(0, round((left - pad_x) * width))
        y1 = max(0, round((top - pad_y) * height))
        x2 = min(width, round((right + pad_x) * width))
        y2 = min(height, round((bottom + pad_y) * height))
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = registered[y1:y2, x1:x2]
    return Image.fromarray(result, "RGB")


def rescue_colored_marks(source: Image.Image, candidate: Image.Image) -> Image.Image:
    """Preserve large pen/stamp strokes without restoring printed page damage."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size:
        return candidate
    pixels = np.asarray(source)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = ((saturation >= 55) & (value <= 225)).astype(np.uint8)
    colored = np.asarray(
        cv2.morphologyEx(
            colored,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
            iterations=1,
        ),
        dtype=np.uint8,
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(colored, connectivity=8)
    regions: list[tuple[float, float, float, float]] = []
    width, height = source.size
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        safely_interior = (
            x > width * 0.02
            and y > height * 0.02
            and x + component_width < width * 0.98
            and y + component_height < height * 0.98
        )
        if (
            safely_interior
            and component_width >= width * 0.025
            and component_height >= height * 0.02
            and area >= max(24, round(width * height * 0.00002))
        ):
            regions.append(
                (
                    x / width,
                    y / height,
                    (x + component_width) / width,
                    (y + component_height) / height,
                )
            )
    return restore_source_regions(source, candidate, regions)


def rescue_edge_text(source: Image.Image, candidate: Image.Image, *, language: str) -> Image.Image:
    """Restore tiny edge text while leaving holes, dirt, and page borders removed."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size:
        return candidate
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    candidate_to_source = _registration_matrix(source_gray, candidate_gray)
    if candidate_to_source is None:
        return candidate

    tokens = ocr_tokens(source, language)
    candidate_tokens = ocr_tokens(candidate, language)
    unsafe_edges = {
        edge: edge_tokens
        for edge in ("left", "top", "right", "bottom")
        if (edge_tokens := _outer_microprint(tokens, edge))
        and not _edge_model_is_authoritative(
            edge_tokens,
            candidate_tokens,
            candidate_to_source,
            edge=edge,
        )
    }
    if not unsafe_edges:
        return candidate

    width, height = source.size
    mask = np.zeros((height, width), dtype=np.uint8)
    erase = np.zeros_like(mask)
    for edge_tokens in unsafe_edges.values():
        # Restore one continuous registered strip. Besides avoiding seams between OCR
        # boxes, this preserves punctuation and marks that OCR did not tokenize.
        _paint_region(
            mask,
            _token_region(
                edge_tokens,
                horizontal_padding=1,
                vertical_padding=1,
            ),
        )
        # Clear a larger, OCR-derived neighborhood before compositing so locally
        # displaced or duplicated model text cannot remain around the exact source.
        _paint_region(
            erase,
            _token_region(
                edge_tokens,
                horizontal_padding=12,
                vertical_padding=12,
            ),
        )

    source_to_candidate = np.linalg.inv(candidate_to_source)
    matrix = _pixel_matrix(source_to_candidate, width, height)
    warped_source = cv2.warpPerspective(
        _clean_source(source),
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_source = _remove_paper_tone(warped_source)
    warped_source = _remove_scanner_borders(warped_source)
    warped_mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    warped_erase = cv2.warpPerspective(erase, matrix, (width, height), flags=cv2.INTER_NEAREST)
    result = np.asarray(candidate).copy()
    result[warped_erase > 0] = 255
    result[warped_mask > 0] = warped_source[warped_mask > 0]
    return Image.fromarray(result, "RGB")


def best_repair_region(
    discrepancies: Sequence[Discrepancy],
) -> tuple[float, float, float, float] | None:
    repairable = [item for item in discrepancies if item.category in REPAIRABLE_CATEGORIES]
    if not repairable:
        return None
    selected = max(repairable, key=lambda item: SEVERITY_RANK.get(item.severity, 0))
    left, top, right, bottom = selected.region
    pad_x = max(0.02, (right - left) * 0.25)
    pad_y = max(0.02, (bottom - top) * 0.5)
    return (
        max(0.0, left - pad_x),
        max(0.0, top - pad_y),
        min(1.0, right + pad_x),
        min(1.0, bottom + pad_y),
    )


def repair_region(
    source: Image.Image,
    candidate: Image.Image,
    region: tuple[float, float, float, float],
    *,
    client: ImageGenerator,
    max_edge: int,
) -> Image.Image:
    """Regenerate one reviewer-identified crop at high resolution and splice it back."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    candidate_to_source = _registration_matrix(source_gray, candidate_gray)
    if candidate_to_source is None:
        return candidate

    left, top, right, bottom = region
    source_points: list[np.ndarray] = []
    for x, y in ((left, top), (right, top), (left, bottom), (right, bottom)):
        point = candidate_to_source @ np.array([x, y, 1.0], dtype=np.float64)
        if abs(point[2]) < 1e-9:
            return candidate
        source_points.append(point[:2] / point[2])
    x1 = max(0, round(min(point[0] for point in source_points) * source.width))
    y1 = max(0, round(min(point[1] for point in source_points) * source.height))
    x2 = min(source.width, round(max(point[0] for point in source_points) * source.width))
    y2 = min(source.height, round(max(point[1] for point in source_points) * source.height))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return candidate
    crop = source.crop((x1, y1, x2, y2))
    scale = min(4.0, max_edge / max(crop.size))
    if scale > 1:
        crop = crop.resize(
            (round(crop.width * scale), round(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
    generated = client.generate(crop, REGIONAL_REPAIR_PROMPT, max_edge=max_edge)
    repaired = finish_pristine_recreation(
        normalize_generated(generated, crop.size, source_dpi=300).image
    )

    dx1, dy1, dx2, dy2 = (
        round(left * candidate.width),
        round(top * candidate.height),
        round(right * candidate.width),
        round(bottom * candidate.height),
    )
    repaired = repaired.resize((dx2 - dx1, dy2 - dy1), Image.Resampling.LANCZOS)
    result = candidate.copy()
    result.paste(repaired, (dx1, dy1))
    return result
