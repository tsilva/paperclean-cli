"""Pristine reconstruction helpers that preserve hard-to-read source content."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageFilter

from paperclean.imaging import (
    finish_pristine_recreation,
    normalize_generated,
    review_view_pairs,
)
from paperclean.models import Discrepancy, PageGeometry
from paperclean.prompting import REGIONAL_REPAIR_PROMPT
from paperclean.validation import _registration_matrix

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
    "other_content",
}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class ImageGenerator(Protocol):
    def generate(self, source: Image.Image, prompt: str, *, max_edge: int) -> Image.Image: ...


@dataclass(frozen=True, slots=True)
class PagePlane:
    corners: np.ndarray
    area_fraction: float
    confidence: float


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [
            points[np.argmin(total)],
            points[np.argmin(difference)],
            points[np.argmax(total)],
            points[np.argmax(difference)],
        ],
        dtype=np.float32,
    )


def detect_page_plane(source: Image.Image) -> PagePlane | None:
    """Find a dominant photographed sheet without assuming a particular page size."""
    pixels = np.asarray(source.convert("RGB"))
    height, width = pixels.shape[:2]
    scale = min(1.0, 1600 / max(width, height))
    work = (
        cv2.resize(
            pixels,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1
        else pixels
    )
    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0]
    chroma = np.abs(lab[:, :, 1].astype(np.int16) - 128) + np.abs(
        lab[:, :, 2].astype(np.int16) - 128
    )
    background = cv2.GaussianBlur(lightness, (0, 0), sigmaX=max(work.shape[:2]) / 18)
    paper = ((lightness >= background - 12) & (lightness >= 85) & (chroma <= 72)).astype(np.uint8)
    close_size = max(7, round(min(work.shape[:2]) * 0.022))
    if close_size % 2 == 0:
        close_size += 1
    paper = np.asarray(
        cv2.morphologyEx(
            paper,
            cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), np.uint8),
        ),
        dtype=np.uint8,
    )
    contours, _hierarchy = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas_area = work.shape[0] * work.shape[1]
    candidates: list[PagePlane] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < canvas_area * 0.22:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, 0.025 * perimeter, True)
        if len(polygon) == 4 and cv2.isContourConvex(polygon):
            corners = polygon.reshape(4, 2).astype(np.float32)
        else:
            corners = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        corners = _order_corners(corners)
        polygon_area = abs(float(cv2.contourArea(corners)))
        if polygon_area < canvas_area * 0.25:
            continue
        fill = min(1.0, contour_area / max(polygon_area, 1.0))
        normalized = corners / scale
        area_fraction = polygon_area / canvas_area
        # Prefer a large, solid light sheet; confidence is independent of document text.
        confidence = min(1.0, fill * 0.7 + min(1.0, area_fraction / 0.65) * 0.3)
        candidates.append(PagePlane(normalized, area_fraction, confidence))
    return max(candidates, key=lambda item: item.confidence) if candidates else None


def rectify_photographed_page(source: Image.Image) -> Image.Image | None:
    """Crop and perspective-flatten a confidently detected photographed sheet."""
    plane = detect_page_plane(source)
    if plane is None or plane.confidence < 0.72 or plane.area_fraction > 0.98:
        return None
    top_left, top_right, bottom_right, bottom_left = plane.corners
    target_width = round(
        max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
    )
    target_height = round(
        max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
    )
    if target_width < 64 or target_height < 64:
        return None
    destination = np.asarray(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(plane.corners.astype(np.float32), destination)
    rectified = cv2.warpPerspective(
        np.asarray(source.convert("RGB")),
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rectified, "RGB")


def _robust_boundary_curve(
    positions: np.ndarray,
    values: np.ndarray,
    *,
    extent: int,
    degree: int,
) -> np.ndarray | None:
    """Fit a smooth page boundary while rejecting fingers and concave notches."""
    if len(positions) < max(32, round(extent * 0.45)):
        return None
    center = (extent - 1) / 2
    scale = max(1.0, center)
    normalized = (positions.astype(np.float64) - center) / scale
    keep = np.ones(len(positions), dtype=bool)
    coefficients: np.ndarray | None = None
    for _iteration in range(6):
        if int(np.count_nonzero(keep)) < degree + 2:
            return None
        coefficients = np.polyfit(normalized[keep], values[keep], degree)
        predicted = np.polyval(coefficients, normalized)
        residual = values - predicted
        residual_center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - residual_center))) + 1e-6
        threshold = max(4.0, extent * 0.002, 3.5 * 1.4826 * mad)
        next_keep = np.abs(residual - residual_center) <= threshold
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    if coefficients is None or float(np.mean(keep)) < 0.55:
        return None
    all_positions = np.arange(extent, dtype=np.float64)
    return np.polyval(coefficients, (all_positions - center) / scale).astype(np.float32)


def _straighten_surface_axis(
    pixels: np.ndarray,
    page_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Straighten the first/last mask pixel in every column of one image axis."""
    height, width = page_mask.shape
    positions: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    minimum_span = height * 0.25
    for x in range(width):
        rows = np.flatnonzero(page_mask[:, x] > 0)
        if len(rows) < minimum_span:
            continue
        positions.append(x)
        starts.append(float(rows[0]))
        ends.append(float(rows[-1]))
    position_array = np.asarray(positions, dtype=np.float64)
    start_curve = _robust_boundary_curve(
        position_array,
        np.asarray(starts, dtype=np.float64),
        extent=width,
        degree=4,
    )
    end_curve = _robust_boundary_curve(
        position_array,
        np.asarray(ends, dtype=np.float64),
        extent=width,
        degree=3,
    )
    if start_curve is None or end_curve is None:
        return pixels, page_mask, False
    span = end_curve - start_curve
    if np.any(~np.isfinite(span)) or float(np.percentile(span, 5)) < height * 0.30:
        return pixels, page_mask, False
    boundary_variation = max(float(np.ptp(start_curve)), float(np.ptp(end_curve)))
    if boundary_variation < height * 0.006:
        return pixels, page_mask, False

    target_margin = max(2, round(height * 0.025))
    target_start = float(target_margin)
    target_end = float(height - 1 - target_margin)
    output_rows = np.arange(height, dtype=np.float32)
    fraction = np.clip(
        (output_rows[:, None] - target_start) / max(1.0, target_end - target_start),
        0.0,
        1.0,
    )
    map_y = start_curve[None, :] + fraction * span[None, :]
    map_x = np.broadcast_to(
        np.arange(width, dtype=np.float32)[None, :],
        (height, width),
    ).copy()
    remapped_pixels = cv2.remap(
        pixels,
        map_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    remapped_mask = cv2.remap(
        page_mask,
        map_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    outside = (output_rows < target_start) | (output_rows > target_end)
    remapped_pixels[outside, :] = 255
    remapped_mask[outside, :] = 0
    return remapped_pixels, remapped_mask, True


def dewarp_curved_page(image: Image.Image, page_mask: np.ndarray) -> Image.Image:
    """Flatten residual two-axis sheet curl using the rectified physical-page mask."""
    pixels = np.asarray(image.convert("RGB")).copy()
    if page_mask.shape != pixels.shape[:2]:
        return image.convert("RGB")
    mask = np.asarray(page_mask > 0, dtype=np.uint8) * 255
    pixels, mask, _vertical_changed = _straighten_surface_axis(pixels, mask)
    transposed_pixels = np.transpose(pixels, (1, 0, 2)).copy()
    transposed_mask = mask.T.copy()
    transposed_pixels, transposed_mask, _horizontal_changed = _straighten_surface_axis(
        transposed_pixels,
        transposed_mask,
    )
    pixels = np.transpose(transposed_pixels, (1, 0, 2)).copy()
    return Image.fromarray(pixels, "RGB")


def _rectify_page_geometry_with_mask(
    source: Image.Image,
    geometry: PageGeometry,
) -> tuple[Image.Image, np.ndarray] | None:
    """Flatten model-located page corners and retain its visible-paper mask."""
    source = source.convert("RGB")
    if geometry.confidence < 0.70:
        return None
    width, height = source.size
    source_pixels = np.asarray(source).copy()
    page_polygon = geometry.page_polygon or geometry.corners
    source_page_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(
        source_page_mask,
        [
            np.asarray(
                [(x * width, y * height) for x, y in page_polygon],
                dtype=np.int32,
            )
        ],
        255,
    )
    visible_page_mask = source_page_mask.copy()
    # Stay just inside segmentation antialiasing without consuming the document's
    # real margin. Low-frequency curl and lighting shadows belong to the later
    # photometric cleanup; using a large geometry erosion here can clip letterheads,
    # signatures, stamps, and handwritten notes close to the physical sheet edge.
    # Erosion removes approximately half the kernel width, hence this 4% kernel is
    # a roughly 2% physical safety inset.
    edge_inset = max(3, round(min(width, height) * 0.04))
    if edge_inset % 2 == 0:
        edge_inset += 1
    source_page_mask = np.asarray(
        cv2.erode(
            source_page_mask,
            np.ones((edge_inset, edge_inset), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
        dtype=np.uint8,
    )
    edge_content_mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in geometry.edge_content:
        points = np.asarray([(x * width, y * height) for x, y in polygon], dtype=np.int32)
        if len(points) >= 4:
            cv2.fillPoly(edge_content_mask, [points], 255)
    source_float = source_pixels.astype(np.float32)
    evidence_background = cv2.GaussianBlur(
        source_float,
        (0, 0),
        sigmaX=max(width, height) / 28,
        sigmaY=max(width, height) / 28,
    )
    evidence_normalized = np.clip(
        source_float * 246 / np.maximum(evidence_background, 36), 0, 255
    ).astype(np.uint8)
    evidence_gray = cv2.cvtColor(evidence_normalized, cv2.COLOR_RGB2GRAY)
    evidence_chroma = evidence_normalized.max(axis=2).astype(np.int16) - evidence_normalized.min(
        axis=2
    ).astype(np.int16)
    authored_evidence = (
        (evidence_gray < 215) | ((evidence_chroma > 10) & (evidence_gray < 252))
    ).astype(np.uint8)
    evidence_support = max(3, round(min(width, height) * 0.002))
    authored_evidence = cv2.dilate(
        authored_evidence,
        np.ones((evidence_support, evidence_support), np.uint8),
    )
    source_page_mask[
        (edge_content_mask > 0) & (authored_evidence > 0) & (visible_page_mask > 0)
    ] = 255
    for polygon in geometry.occlusions:
        points = np.asarray([(x * width, y * height) for x, y in polygon], dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(source_pixels, [points], (255, 255, 255))
            cv2.fillPoly(source_page_mask, [points], 0)
    source_pixels[source_page_mask == 0] = 255
    corners = np.asarray([(x * width, y * height) for x, y in geometry.corners], dtype=np.float32)
    if not np.all(np.isfinite(corners)) or not cv2.isContourConvex(corners.astype(np.int32)):
        return None
    area_fraction = abs(float(cv2.contourArea(corners))) / (width * height)
    if not 0.20 <= area_fraction <= 1.02:
        return None
    top_left, top_right, bottom_right, bottom_left = corners
    page_width = max(
        np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)
    )
    page_height = max(
        np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)
    )
    if page_width < 64 or page_height < 64:
        return None
    margin = max(2, round(min(width, height) * 0.012))
    scale = min((width - 2 * margin) / page_width, (height - 2 * margin) / page_height)
    target_width = max(1, round(page_width * scale))
    target_height = max(1, round(page_height * scale))
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    destination = np.asarray(
        [
            [left, top],
            [left + target_width - 1, top],
            [left + target_width - 1, top + target_height - 1],
            [left, top + target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    rectified = cv2.warpPerspective(
        source_pixels,
        matrix,
        source.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    rectified_source_mask = cv2.warpPerspective(
        source_page_mask,
        matrix,
        source.size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectified_visible_mask = cv2.warpPerspective(
        visible_page_mask,
        matrix,
        source.size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    page_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(page_mask, destination.astype(np.int32), 255)
    rectified[(page_mask == 0) | (rectified_source_mask == 0)] = 255
    rectified_visible_mask[page_mask == 0] = 0
    return Image.fromarray(rectified, "RGB"), rectified_visible_mask


def rectify_page_geometry(source: Image.Image, geometry: PageGeometry) -> Image.Image | None:
    """Flatten and surface-dewarp a model-located page using only source pixels."""
    result = _rectify_page_geometry_with_mask(source, geometry)
    if result is None:
        return None
    rectified, page_mask = result
    return dewarp_curved_page(rectified, page_mask)


def remove_photo_capture_artifacts(image: Image.Image) -> Image.Image:
    """Whiten broad non-paper camera artifacts while retaining thin authored marks."""
    pixels = np.asarray(image.convert("RGB")).copy()
    height, width = pixels.shape[:2]
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    saturation = hsv[:, :, 1]
    # Camera surroundings, fingers, and clothing form broad dark or saturated
    # regions. Printed/handwritten ink is thin, so a large opening removes it from
    # the artifact mask before any pixels are whitened.
    artifact_seed = ((gray < 105) | ((saturation > 75) & (gray < 235))).astype(np.uint8)
    kernel_size = max(7, round(min(width, height) * 0.012))
    if kernel_size % 2 == 0:
        kernel_size += 1
    broad = cv2.morphologyEx(
        artifact_seed,
        cv2.MORPH_OPEN,
        np.ones((kernel_size, kernel_size), np.uint8),
    )
    close_size = max(kernel_size, round(min(width, height) * 0.028))
    if close_size % 2 == 0:
        close_size += 1
    broad = cv2.morphologyEx(
        broad,
        cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(broad, connectivity=8)
    page_area = width * height
    removal = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x <= width * 0.03
            or y <= height * 0.03
            or x + component_width >= width * 0.97
            or y + component_height >= height * 0.97
        )
        if area >= page_area * 0.0005 and touches_border:
            removal[labels == label] = 255
    dilation = max(3, round(min(width, height) * 0.004))
    removal = np.asarray(
        cv2.dilate(removal, np.ones((dilation, dilation), np.uint8)), dtype=np.uint8
    )
    pixels[removal > 0] = 255
    return Image.fromarray(pixels, "RGB")


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


def rectify_source_to_reference(source: Image.Image, reference: Image.Image) -> Image.Image | None:
    """Warp exact source pixels onto a model-proposed flattened page lattice.

    The reference supplies geometry only. Its generated text and marks are never
    copied, which lets the model isolate a difficult photographed page without being
    trusted to redraw handwriting or microprint.
    """
    source = source.convert("RGB")
    reference = reference.convert("RGB")
    if source.size != reference.size:
        return None
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(np.asarray(reference), cv2.COLOR_RGB2GRAY)
    reference_to_source = _registration_matrix(source_gray, reference_gray)
    if reference_to_source is None:
        return None
    try:
        source_to_reference = np.linalg.inv(reference_to_source)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(source_to_reference)):
        return None
    matrix = _pixel_matrix(source_to_reference, source.width, source.height)
    warped = cv2.warpPerspective(
        np.asarray(source),
        matrix,
        reference.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(warped, "RGB")


def align_candidate_to_source(source: Image.Image, candidate: Image.Image) -> Image.Image | None:
    """Place a generated candidate onto an already rectified source coordinate lattice."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size:
        return None
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    candidate_to_source = _registration_matrix(source_gray, candidate_gray)
    if candidate_to_source is None or not np.all(np.isfinite(candidate_to_source)):
        return None
    matrix = _pixel_matrix(candidate_to_source, source.width, source.height)
    aligned = cv2.warpPerspective(
        np.asarray(candidate),
        matrix,
        source.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(aligned, "RGB")


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
    """Remove boundary rails and punched holes while retaining nearby small ink."""
    result = pixels.copy()
    height, width = pixels.shape[:2]
    edge = max(1, round(width * 0.012))
    result[:, :edge] = 255
    result[:, -edge:] = 255
    dark = (cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) < 140).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for label in range(1, count):
        x, y, component_width, component_height, _area = stats[label]
        touches_vertical_edge = x < width * 0.012 or x + component_width > width * 0.988
        touches_horizontal_edge = y < height * 0.012 or y + component_height > height * 0.988
        vertical_rail = (
            touches_vertical_edge
            and component_height > height * 0.04
            and component_height > component_width * 4
        )
        horizontal_rail = (
            touches_horizontal_edge
            and component_width > width * 0.04
            and component_width > component_height * 4
        )
        if vertical_rail or horizontal_rail:
            component = (labels == label).astype(np.uint8)
            short_edge = min(width, height)
            dilation = max(3, round(short_edge * 0.004))
            component = cv2.dilate(component, np.ones((dilation, dilation), np.uint8))
            result[component > 0] = 255
    return _remove_punch_holes(result)


def _remove_low_information_edge_artifacts(pixels: np.ndarray) -> np.ndarray:
    """Whiten shallow, low-information scanner/capture remnants at page edges."""
    result = pixels.copy()
    height, width = result.shape[:2]
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    # Cleanup has already mapped ordinary paper to pure white.  Join only the
    # remaining near-edge non-white fragments so a mottled scanner-bed wedge is
    # considered as one component instead of thousands of isolated speckles.
    seed = (gray < 245).astype(np.uint8)
    close_y = max(3, round(height * 0.004))
    close_x = max(3, round(width * 0.004))
    joined = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        np.ones((close_y, close_x), np.uint8),
    )
    open_y = max(3, round(height * 0.002))
    open_x = max(3, round(width * 0.002))
    broad = cv2.morphologyEx(
        joined,
        cv2.MORPH_OPEN,
        np.ones((open_y, open_x), np.uint8),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        broad,
        connectivity=8,
    )
    minimum_area = height * width * 0.0001
    removal = np.zeros((height, width), dtype=np.uint8)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        if area < minimum_area:
            continue
        touches_left = x <= width * 0.015
        touches_top = y <= height * 0.015
        touches_right = x + component_width >= width * 0.985
        touches_bottom = y + component_height >= height * 0.985
        shallow = (
            ((touches_top or touches_bottom) and component_height <= height * 0.08)
            or ((touches_left or touches_right) and component_width <= width * 0.08)
        )
        if not shallow:
            continue
        component = labels == label
        dark_fraction = float(np.mean(gray[component] < 180))
        if dark_fraction >= 0.25:
            # Edge-touching text, rules, barcodes, and image panels contain a
            # meaningful dark core.  A pale scanner-bed shadow does not.
            continue
        component_y, component_x = np.where(component)
        if component_x.size < 3:
            continue
        hull = cv2.convexHull(
            np.column_stack((component_x, component_y)).astype(np.int32)
        )
        component_mask = np.zeros_like(removal)
        cv2.fillConvexPoly(component_mask, hull, 255)
        expansion = max(3, round(min(width, height) * 0.004))
        component_mask = np.asarray(
            cv2.dilate(
                component_mask,
                np.ones((expansion, expansion), np.uint8),
            ),
            dtype=np.uint8,
        )
        removal[component_mask > 0] = 255
    result[removal > 0] = 255
    return result


def _remove_punch_holes(pixels: np.ndarray) -> np.ndarray:
    """Erase dark circular binder holes only within the outer side margins."""
    result = pixels.copy()
    for center_x, center_y, radius, padding, touches_authored_ink in _punch_hole_candidates(result):
        if touches_authored_ink:
            continue
        cv2.circle(result, (center_x, center_y), radius + padding, (255, 255, 255), thickness=-1)
    return result


def _punch_hole_candidates(
    pixels: np.ndarray,
) -> list[tuple[int, int, int, int, bool]]:
    """Return filled side-margin circles and whether they connect to authored ink."""
    height, width = pixels.shape[:2]
    short_edge = min(width, height)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    min_radius = max(3, round(short_edge * 0.008))
    max_radius = max(4, round(short_edge * 0.035))
    padding = max(2, round(short_edge * 0.002))
    photographic_mask = _photographic_region_mask(pixels)

    authored = (gray < 180).astype(np.uint8)
    _authored_count, authored_labels, authored_stats, authored_centroids = (
        cv2.connectedComponentsWithStats(authored, connectivity=8)
    )

    def candidate_padding(radius: int) -> int:
        return int(max(padding, round(radius * 0.40)))

    def in_side_margin(center_x: int, radius: int, erase_padding: int) -> bool:
        return bool(
            center_x + radius + erase_padding <= width * 0.10
            or center_x - radius - erase_padding >= width * 0.90
        )

    def touches_authored_ink(
        center_x: int,
        center_y: int,
        radius: int,
        erase_padding: int,
    ) -> bool:
        if not (0 <= center_x < width and 0 <= center_y < height):
            return True
        hole_label = int(authored_labels[center_y, center_x])
        if hole_label == 0:
            disk = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(disk, (center_x, center_y), radius, 255, thickness=-1)
            labels_in_disk = authored_labels[(disk > 0) & (authored_labels > 0)]
            if labels_in_disk.size:
                hole_label = int(np.bincount(labels_in_disk).argmax())
        extent = radius + erase_padding
        erase_disk = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(erase_disk, (center_x, center_y), extent, 255, thickness=-1)
        nearby_labels = np.unique(authored_labels[(erase_disk > 0) & (authored_labels > 0)])
        for nearby_label in nearby_labels:
            x, y, component_width, component_height, area = authored_stats[nearby_label]
            if int(nearby_label) != hole_label and area < max(4, round(radius**2 * 0.02)):
                continue
            contained = (
                x >= center_x - extent
                and y >= center_y - extent
                and x + component_width <= center_x + extent + 1
                and y + component_height <= center_y + extent + 1
            )
            component_x, component_y = authored_centroids[nearby_label]
            centered_hole_evidence = (
                np.hypot(component_x - center_x, component_y - center_y) <= radius * 0.25
            )
            if not contained or (int(nearby_label) != hole_label and not centered_hole_evidence):
                return True
        return False

    candidates: list[tuple[int, int, int, int, bool]] = []

    dark = (gray < 140).astype(np.uint8)
    component_count, _component_labels, component_stats, component_centroids = (
        cv2.connectedComponentsWithStats(dark, connectivity=8)
    )
    min_component_area = np.pi * min_radius**2 * 0.50
    max_component_area = np.pi * max_radius**2 * 1.50
    for label in range(1, component_count):
        _x, _y, component_width, component_height, area = component_stats[label]
        if component_width == 0 or component_height == 0:
            continue
        aspect = component_width / component_height
        density = area / (component_width * component_height)
        center_x, center_y = np.rint(component_centroids[label]).astype(int)
        radius = round((component_width + component_height) / 4)
        erase_padding = candidate_padding(radius)
        if (
            min_component_area <= area <= max_component_area
            and 0.65 <= aspect <= 1.55
            and density >= 0.68
            and min_radius <= radius <= max_radius
            and in_side_margin(int(center_x), radius, padding)
            and not photographic_mask[int(center_y), int(center_x)]
        ):
            candidates.append(
                (
                    int(center_x),
                    int(center_y),
                    radius,
                    erase_padding,
                    touches_authored_ink(int(center_x), int(center_y), radius, erase_padding),
                )
            )

    # A punch core merged into a text line can make its whole connected component
    # non-circular. Distance-transform maxima still reveal the thick round core,
    # while ordinary glyph strokes remain below the minimum punch radius.
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    local_maxima = (
        (distance == cv2.dilate(distance, np.ones((7, 7), np.uint8)))
        & (distance >= min_radius)
        & (distance <= max_radius)
    )
    maxima_count, _maxima_labels, _maxima_stats, maxima_centroids = (
        cv2.connectedComponentsWithStats(local_maxima.astype(np.uint8), connectivity=8)
    )
    yy, xx = np.ogrid[:height, :width]
    for center_x, center_y in np.rint(maxima_centroids[1:maxima_count]).astype(int):
        if not (0 <= center_x < width and 0 <= center_y < height):
            continue
        radius = round(float(distance[center_y, center_x]))
        erase_padding = candidate_padding(radius)
        if not in_side_margin(center_x, radius, padding):
            continue
        if not (height * 0.08 <= center_y <= height * 0.92):
            continue
        if photographic_mask[center_y, center_x]:
            continue
        if any(
            np.hypot(center_x - existing_x, center_y - existing_y) <= max(radius, existing_radius)
            for existing_x, existing_y, existing_radius, _padding, _touches in candidates
        ):
            continue
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        annulus = (squared_distance > radius**2) & (squared_distance <= (radius * 1.45) ** 2)
        annulus_area = int(np.count_nonzero(annulus))
        if annulus_area == 0:
            continue
        annulus_dark_fraction = int(np.count_nonzero((dark > 0) & annulus)) / annulus_area
        if annulus_dark_fraction > 0.55:
            continue
        candidates.append(
            (
                int(center_x),
                int(center_y),
                radius,
                erase_padding,
                touches_authored_ink(int(center_x), int(center_y), radius, erase_padding),
            )
        )

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, round(short_edge * 0.04)),
        param1=100,
        param2=14,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return candidates
    for center_x, center_y, radius in np.rint(circles[0]).astype(int):
        if not (0 <= center_x < width and 0 <= center_y < height):
            continue
        erase_padding = candidate_padding(int(radius))
        if not in_side_margin(int(center_x), int(radius), padding):
            continue
        if photographic_mask[int(center_y), int(center_x)]:
            continue
        if any(
            np.hypot(center_x - existing_x, center_y - existing_y) <= max(radius, existing_radius)
            for existing_x, existing_y, existing_radius, _padding, _touches in candidates
        ):
            continue
        disk = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(disk, (center_x, center_y), radius, 255, thickness=-1)
        disk_area = int(np.count_nonzero(disk))
        if disk_area == 0:
            continue
        dark_fill = int(np.count_nonzero((gray < 140) & (disk > 0))) / disk_area
        if dark_fill < 0.70:
            continue
        candidates.append(
            (
                int(center_x),
                int(center_y),
                int(radius),
                erase_padding,
                touches_authored_ink(int(center_x), int(center_y), int(radius), erase_padding),
            )
        )
    return candidates


def authored_punch_hole_regions(
    source: Image.Image,
) -> list[tuple[float, float, float, float]]:
    """Find punch holes that obscure nearby authored ink and merit cautious restoration."""
    pixels = _remove_paper_tone(_clean_source(source.convert("RGB")))
    height, width = pixels.shape[:2]
    regions: list[tuple[float, float, float, float]] = []
    for center_x, center_y, radius, padding, touches_authored_ink in _punch_hole_candidates(pixels):
        if not touches_authored_ink:
            continue
        vertical_context = max(radius * 3, round(height * 0.025))
        if center_x < width / 2:
            x1 = max(0, center_x - radius - padding)
            x2 = min(width, center_x + radius + padding + round(width * 0.28))
        else:
            x1 = max(0, center_x - radius - padding - round(width * 0.28))
            x2 = min(width, center_x + radius + padding)
        y1 = max(0, center_y - vertical_context)
        y2 = min(height, center_y + vertical_context)
        regions.append((x1 / width, y1 / height, x2 / width, y2 / height))
    return regions


def _page_skew_angle(pixels: np.ndarray) -> float | None:
    """Estimate a small global page rotation from long near-horizontal evidence."""
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    max_side = 1600
    scale = min(1.0, max_side / max(gray.shape))
    if scale < 1:
        size = (round(gray.shape[1] * scale), round(gray.shape[0] * scale))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800,
        threshold=max(40, gray.shape[1] // 8),
        minLineLength=gray.shape[1] // 6,
        maxLineGap=max(10, gray.shape[1] // 50),
    )
    if lines is None:
        return None
    angles = [
        float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
        if abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))) <= 5
    ]
    if len(angles) < 4:
        return None
    angle = float(np.median(angles))
    return angle if 0.15 <= abs(angle) <= 3.0 else None


def _page_rotation_matrix(pixels: np.ndarray, angle: float) -> np.ndarray:
    height, width = pixels.shape[:2]
    radians = np.radians(angle)
    cosine = abs(float(np.cos(radians)))
    sine = abs(float(np.sin(radians)))
    bound_width = width * cosine + height * sine
    bound_height = width * sine + height * cosine
    contain_scale = min(width / bound_width, height / bound_height)
    return cv2.getRotationMatrix2D((width / 2, height / 2), angle, contain_scale)


def _rotate_page(pixels: np.ndarray, angle: float | None) -> np.ndarray:
    if angle is None:
        return pixels
    height, width = pixels.shape[:2]
    return cv2.warpAffine(
        pixels,
        _page_rotation_matrix(pixels, angle),
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _deskew_page(pixels: np.ndarray) -> np.ndarray:
    return _rotate_page(pixels, _page_skew_angle(pixels))


def _photographic_region_mask(pixels: np.ndarray) -> np.ndarray:
    """Locate large raster or shaded form panels that must stay untouched."""
    height, width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    dark = (gray < 170).astype(np.uint8)
    kernel = np.ones(
        (max(3, round(height * 0.002)), max(3, round(width * 0.003))),
        np.uint8,
    )
    connected = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        connected,
        connectivity=8,
    )
    result = np.zeros((height, width), dtype=np.uint8)
    page_area = width * height
    for label in range(1, count):
        x, y, component_width, component_height, _area = stats[label]
        box_area = component_width * component_height
        if (
            component_width < width * 0.15
            or component_height < height * 0.08
            or box_area < page_area * 0.02
            or component_width > width * 0.95
            or component_height > height * 0.90
        ):
            continue
        density = (
            float(np.count_nonzero(dark[y : y + component_height, x : x + component_width]))
            / box_area
        )
        if density < 0.22:
            continue
        padding_x = max(2, round(width * 0.003))
        padding_y = max(2, round(height * 0.003))
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(width, x + component_width + padding_x)
        y2 = min(height, y + component_height + padding_y)
        result[y1:y2, x1:x2] = 255

    # Pale reference/result panels can carry intentional shading that normalization
    # would otherwise posterize. A large-kernel opening removes text and form rules,
    # leaving only broad regions that are consistently darker than the page stock.
    background_level = float(np.percentile(gray, 75))
    # Use a modest contrast threshold so a single intentionally shaded form
    # column remains connected even when scanner illumination makes its upper
    # and lower halves differ by several gray levels.
    shaded = (gray < background_level - 10).astype(np.uint8)
    shade_kernel = np.ones(
        (max(5, round(height * 0.01)), max(5, round(width * 0.01))),
        np.uint8,
    )
    broad_shading = cv2.morphologyEx(shaded, cv2.MORPH_OPEN, shade_kernel)
    broad_shading = cv2.morphologyEx(
        broad_shading,
        cv2.MORPH_CLOSE,
        np.ones(
            (
                max(5, round(height * 0.025)),
                max(5, round(width * 0.012)),
            ),
            np.uint8,
        ),
    )
    shade_count, shade_labels, shade_stats, _shade_centroids = cv2.connectedComponentsWithStats(
        broad_shading, connectivity=8
    )
    for label in range(1, shade_count):
        x, y, component_width, component_height, area = shade_stats[label]
        box_area = component_width * component_height
        if (
            component_width < width * 0.12
            or component_height < height * 0.08
            or box_area < page_area * 0.02
            or component_width > width * 0.95
            or component_height > height * 0.90
            or area / box_area < 0.65
        ):
            continue
        # Broad low-detail shading connected to a page edge is characteristic of
        # scanner-bed shadows, lifted/curled paper edges, and capture occlusions.
        # Intentional shaded panels (forms, charts, radiographs) contain materially
        # more internal edges.  Do not preserve edge-connected low-information
        # regions, otherwise the cleanup faithfully pastes the scanner artifact
        # back onto an otherwise white page.
        touches_page_edge = (
            x <= width * 0.015
            or y <= height * 0.015
            or x + component_width >= width * 0.985
            or y + component_height >= height * 0.985
        )
        if touches_page_edge:
            component_gray = gray[y : y + component_height, x : x + component_width]
            component_pixels = broad_shading[
                y : y + component_height,
                x : x + component_width,
            ] > 0
            sobel_x = cv2.Sobel(component_gray, cv2.CV_32F, 1, 0)
            sobel_y = cv2.Sobel(component_gray, cv2.CV_32F, 0, 1)
            internal_edge_fraction = float(
                np.mean(np.hypot(sobel_x, sobel_y)[component_pixels] > 40)
            )
            if internal_edge_fraction < 0.06:
                continue
        component = (shade_labels == label).astype(np.uint8)
        if not touches_page_edge and area / box_area >= 0.80:
            # A dense, bounded rectangular component is an authored form panel.
            # Preserve its entire bounding box so illumination gradients cannot
            # split it into polygonal fragments that resemble a redaction.
            component.fill(0)
            component[y : y + component_height, x : x + component_width] = 255
        else:
            component_y, component_x = np.where(component > 0)
            if component_x.size < 3:
                continue
            hull = cv2.convexHull(
                np.column_stack((component_x, component_y)).astype(np.int32)
            )
            component.fill(0)
            cv2.fillConvexPoly(component, hull, 255)
        dilation = max(5, round(min(width, height) * 0.004))
        component = cv2.dilate(component, np.ones((dilation, dilation), np.uint8))
        result[component > 0] = 255
    return result


def has_preserved_photographic_regions(source: Image.Image) -> bool:
    """Report whether source cleanup will preserve large raster or shaded panels."""
    return bool(np.any(_photographic_region_mask(np.asarray(source.convert("RGB")))))


def source_preserving_cleanup(source: Image.Image) -> Image.Image:
    """Whiten and sharpen a scan without synthesizing or rewriting its content."""
    source_pixels = np.asarray(source.convert("RGB"))
    photographic_mask = _photographic_region_mask(source_pixels)
    cleaned = _clean_source(source.convert("RGB"))
    cleaned = _remove_paper_tone(cleaned)
    for center_x, center_y, radius, padding, touches_authored_ink in _punch_hole_candidates(
        cleaned
    ):
        if not touches_authored_ink:
            cv2.circle(
                photographic_mask,
                (center_x, center_y),
                radius + padding,
                0,
                thickness=-1,
            )
    cleaned = _remove_scanner_borders(cleaned)
    angle = _page_skew_angle(cleaned)
    cleaned = _rotate_page(cleaned, angle)
    cleaned = _remove_paper_tone(cleaned)
    if np.any(photographic_mask):
        preserved = _rotate_page(source_pixels, angle)
        if angle is not None:
            height, width = photographic_mask.shape
            photographic_mask = cv2.warpAffine(
                photographic_mask,
                _page_rotation_matrix(source_pixels, angle),
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        cleaned[photographic_mask > 0] = preserved[photographic_mask > 0]
    cleaned = _remove_low_information_edge_artifacts(cleaned)
    return Image.fromarray(cleaned, "RGB")


def photographed_page_cleanup(source: Image.Image) -> Image.Image:
    """Normalize a flattened phone photograph into clean print-like source pixels.

    Unlike scan cleanup, this does not amplify every faint luminance variation: phone
    lighting and paper grain would become visible texture. It estimates illumination,
    retains dark or chromatic authored evidence with a one-pixel antialiasing support,
    and replaces the remaining paper surface with white.
    """
    pixels = np.asarray(source.convert("RGB"), dtype=np.float32)
    width, height = source.size
    estimation_scale = min(1.0, 1600 / max(source.size))
    if estimation_scale < 1.0:
        estimation_size = (
            max(1, round(width * estimation_scale)),
            max(1, round(height * estimation_scale)),
        )
        estimation_pixels = cv2.resize(
            pixels,
            estimation_size,
            interpolation=cv2.INTER_AREA,
        )
    else:
        estimation_size = source.size
        estimation_pixels = pixels
    sigma = max(estimation_size) / 28
    # Rectification puts the retained source plane on a pure-white canvas. A plain
    # Gaussian blur mixes that artificial white with the darker photographed paper
    # near every mask boundary, making broad blue/gray edge shadows look like ink.
    # Estimate illumination with a normalized convolution over source evidence
    # instead. Sparse isolated marks use white as their background so they cannot
    # normalize themselves away.
    source_u8 = estimation_pixels.astype(np.uint8)
    source_gray = cv2.cvtColor(source_u8, cv2.COLOR_RGB2GRAY)
    source_chroma = source_u8.max(axis=2).astype(np.int16) - source_u8.min(axis=2).astype(
        np.int16
    )
    source_support = ((source_gray < 252) | (source_chroma > 4)).astype(np.float32)
    blurred_support = cv2.GaussianBlur(source_support, (0, 0), sigmaX=sigma, sigmaY=sigma)
    weighted_background = cv2.GaussianBlur(
        estimation_pixels * source_support[:, :, None],
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    ) / np.maximum(blurred_support[:, :, None], 0.01)
    background = np.where(
        blurred_support[:, :, None] >= 0.35,
        weighted_background,
        255.0,
    )
    if estimation_size != source.size:
        background = cv2.resize(background, source.size, interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(pixels * 246 / np.maximum(background, 36), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    chroma = normalized.max(axis=2).astype(np.int16) - normalized.min(axis=2).astype(np.int16)
    estimation_gray = (
        cv2.resize(gray, estimation_size, interpolation=cv2.INTER_AREA)
        if estimation_size != source.size
        else gray
    )
    local_scale = max(estimation_size) / 180
    local_background_estimate = cv2.GaussianBlur(
        estimation_gray.astype(np.float32),
        (0, 0),
        sigmaX=local_scale,
        sigmaY=local_scale,
    )
    local_background = (
        cv2.resize(local_background_estimate, source.size, interpolation=cv2.INTER_LINEAR)
        if estimation_size != source.size
        else local_background_estimate
    )
    local_contrast = local_background - gray
    authored = (
        (gray < 120)
        | ((chroma > 16) & (gray < 225) & (local_contrast > 6))
    ).astype(np.uint8)
    # Remove isolated camera noise but retain small punctuation and antialiased edges
    # around genuine connected strokes.
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(authored, connectivity=8)
    page_area = gray.size
    retained = np.zeros_like(authored)
    for label in range(1, count):
        _x, _y, component_width, component_height, area = stats[label]
        if (
            area >= max(3, round(page_area * 0.0000005))
            or max(component_width, component_height) >= 4
        ):
            retained[labels == label] = 1
    # Retain the immediate antialiasing edge, not a photographed halo around each
    # stroke. Faint semantic marks are recovered separately from reviewer regions.
    support_width = max(3, round(min(source.size) * 0.002))
    if support_width % 2 == 0:
        support_width += 1
    support = cv2.dilate(retained, np.ones((support_width, support_width), np.uint8))
    result = np.full_like(normalized, 255)
    result[support > 0] = normalized[support > 0]
    result = np.asarray(
        Image.fromarray(result, "RGB").filter(
            ImageFilter.UnsharpMask(radius=0.55, percent=120, threshold=2)
        )
    ).copy()
    # Every later capture-artifact deletion must respect authored support. Faint
    # footer microprint may have only a small dark/colored core surrounded by pale
    # antialiasing; deleting its pale edge component independently can erase the
    # complete letter even though the source evidence is unambiguous.
    protected_width = max(3, round(min(source.size) * 0.006))
    if protected_width % 2 == 0:
        protected_width += 1
    authored_support = cv2.dilate(
        retained,
        np.ones((protected_width, protected_width), np.uint8),
    ) > 0
    # Fingers and hands leaking through a concave page mask are characteristically
    # broad, warm-colored blobs at the rectified canvas edge. Remove only connected
    # warm components which touch that edge; red stamps and annotations elsewhere on
    # the sheet remain untouched. A small dilation removes pale skin antialiasing.
    result_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    # Rectification can place an edge-overlapping hand a few percent inside the
    # output canvas, and illumination normalization can desaturate skin to tan.
    # Use a broader warm family and a physical-edge zone, but still require a
    # large connected component so red/brown punctuation and pen strokes survive.
    warm_capture = (
        ((result_hsv[:, :, 0] <= 35) | (result_hsv[:, :, 0] >= 170))
        & (result_hsv[:, :, 1] >= 10)
        & (result_gray < 250)
    ).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        warm_capture,
        connectivity=8,
    )
    height, width = result_gray.shape
    warm_removal = np.zeros_like(warm_capture)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_edge = (
            x <= width * 0.07
            or y <= height * 0.07
            or x + component_width >= width * 0.93
            or y + component_height >= height * 0.93
        )
        component = labels == label
        classic_skin = np.count_nonzero(
            component
            & (result_hsv[:, :, 0] >= 3)
            & (result_hsv[:, :, 0] <= 25)
            & (result_hsv[:, :, 1] >= 25)
        )
        large_desaturated_hand = area >= max(64, round(result_gray.size * 0.0003))
        classic_skin_fragment = classic_skin >= max(32, round(result_gray.size * 0.00005))
        if touches_edge and (large_desaturated_hand or classic_skin_fragment):
            warm_removal[labels == label] = 1
    warm_dilation = max(3, round(min(source.size) * 0.006))
    if warm_dilation % 2 == 0:
        warm_dilation += 1
    warm_removal = np.asarray(
        cv2.dilate(
            warm_removal,
            np.ones((warm_dilation, warm_dilation), np.uint8),
        ),
        dtype=np.uint8,
    )
    cool_authored = (
        (result_hsv[:, :, 0] >= 80)
        & (result_hsv[:, :, 0] <= 145)
        & (result_hsv[:, :, 1] >= 20)
        & (result_gray < 250)
    ).astype(np.uint8)
    cool_support = cv2.dilate(
        cool_authored,
        np.ones((warm_dilation, warm_dilation), np.uint8),
    ) > 0
    # Dilation may touch nearby blue ink or neutral black print. Delete the warm
    # component itself and only its pale halo; do not erase dark non-warm glyphs.
    warm_delete = (
        (warm_removal > 0)
        & (
            ((warm_capture > 0) & (result_gray >= 100))
            | ((result_gray >= 145) & ~cool_support)
        )
    )
    result[warm_delete] = 255
    # Remove the low-saturation tan illumination fringe that remains around a
    # photographed hand after its connected core is deleted. Blue/cyan authored
    # ink and dark glyph cores are outside this chroma/value interval.
    warm_halo = (
        ((result_hsv[:, :, 0] <= 35) | (result_hsv[:, :, 0] >= 170))
        & (result_hsv[:, :, 1] >= 5)
        & (result_hsv[:, :, 1] <= 100)
        & (result_gray >= 135)
        & (result_gray < 254)
    )
    result[warm_halo] = 255
    # Fold shadows are broad, low-chroma regions attached to a page corner. A
    # morphological opening suppresses thin printed/handwritten strokes before
    # connected components are considered. Restricting removal to neutral pixels
    # around the selected component keeps adjacent blue edge writing intact.
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    result_chroma = result.max(axis=2).astype(np.int16) - result.min(axis=2).astype(np.int16)
    neutral_seed = ((result_chroma <= 48) & (result_gray < 245)).astype(np.uint8)
    neutral_opening = max(7, round(min(source.size) * 0.004))
    if neutral_opening % 2 == 0:
        neutral_opening += 1
    neutral_broad = cv2.morphologyEx(
        neutral_seed,
        cv2.MORPH_OPEN,
        np.ones((neutral_opening, neutral_opening), np.uint8),
    )
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        neutral_broad,
        connectivity=8,
    )
    neutral_removal = np.zeros_like(neutral_seed)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_corner = (
            (x <= width * 0.08 or x + component_width >= width * 0.92)
            and (y <= height * 0.08 or y + component_height >= height * 0.92)
        )
        if touches_corner and area >= max(64, round(result_gray.size * 0.00008)):
            neutral_removal[labels == label] = 1
    neutral_dilation = max(3, round(min(source.size) * 0.006))
    if neutral_dilation % 2 == 0:
        neutral_dilation += 1
    neutral_removal = np.asarray(
        cv2.dilate(
            neutral_removal,
            np.ones((neutral_dilation, neutral_dilation), np.uint8),
        ),
        dtype=np.uint8,
    )
    result[(neutral_removal > 0) & (result_chroma <= 65) & (result_gray < 252)] = 255
    # Ink extraction can leave pale antialiasing around camera objects along the
    # crop boundary. Remove only pale connected components that reach the canvas
    # edge; dark and saturated authored strokes are not in this mask.
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    pale = ((result_gray >= 145) & (result_gray < 254)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(pale, connectivity=8)
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        near_edge = (
            x <= width * 0.04
            or y <= height * 0.04
            or x + component_width >= width * 0.96
            or y + component_height >= height * 0.96
        )
        component = labels == label
        if (
            near_edge
            and area >= max(4, round(result_gray.size * 0.000002))
            and not np.any(component & authored_support)
        ):
            result[component] = 255
    # Perspective warping can leave a compact antialiased triangle where an
    # extrapolated sheet corner meets the concave visible-page mask. Such remnants
    # are confined to a canvas corner in both axes; real edge notes normally extend
    # along one edge, and the rectifier adds a white margin before this cleanup.
    nonwhite = (cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) < 252).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(nonwhite, connectivity=8)
    corner_x = width * 0.08
    corner_y = height * 0.08
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        in_left = x < corner_x and x + component_width <= width * 0.15
        in_right = x + component_width > width - corner_x and x >= width * 0.85
        in_top = y < corner_y and y + component_height <= height * 0.15
        in_bottom = y + component_height > height - corner_y and y >= height * 0.85
        density = area / max(1, component_width * component_height)
        compact_corner_remnant = (
            (in_left or in_right)
            and (in_top or in_bottom)
            and area >= max(25, round(result_gray.size * 0.00002))
            # A warped paper/finger triangle fills most of its bounding box.
            # Printed footer glyphs can also sit wholly in a page corner, but their
            # counters and inter-letter spacing make the connected component much
            # sparser. Keeping sparse components prevents the geometric artifact
            # cleanup from erasing real edge microprint.
            and density >= 0.45
        )
        if compact_corner_remnant:
            result[labels == label] = 255
    # Remove pale vertical/horizontal mask rails in a slightly wider outer band,
    # while retaining any genuinely dark or saturated edge handwriting. The final
    # hard-clear is deliberately narrower and handles only interpolation at the
    # literal canvas boundary.
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    outer_band = np.zeros((height, width), dtype=bool)
    band_x = max(1, round(width * 0.03))
    band_y = max(1, round(height * 0.03))
    outer_band[:, :band_x] = True
    outer_band[:, -band_x:] = True
    outer_band[:band_y, :] = True
    outer_band[-band_y:, :] = True
    pale_rail = outer_band & (result_gray >= 145) & ~authored_support
    result[pale_rail] = 255
    return clear_page_border(Image.fromarray(result, "RGB"), fraction=0.012)


def restore_source_regions(
    source: Image.Image,
    candidate: Image.Image,
    regions: Sequence[tuple[float, float, float, float]],
    *,
    already_aligned: bool = False,
) -> Image.Image:
    """Preserve authored source pixels in selected regions on a clean white ground."""
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size or not regions:
        return candidate
    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    matrix = np.eye(3, dtype=np.float64)
    if not already_aligned:
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


def restore_source_evidence_regions(
    source: Image.Image,
    candidate: Image.Image,
    regions: Sequence[tuple[float, float, float, float]],
    *,
    already_aligned: bool = False,
) -> Image.Image:
    """Restore faint source evidence in reviewer-identified semantic regions.

    The source is background-normalized, registered to the candidate, and only
    darker or colored evidence pixels are merged. This avoids copying rectangular
    patches of aged paper while retaining strokes that cleanup made too faint.
    """
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size or not regions:
        return candidate

    source_gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_RGB2GRAY)
    matrix = np.eye(3, dtype=np.float64)
    if not already_aligned:
        candidate_to_source = _registration_matrix(source_gray, candidate_gray)
        if candidate_to_source is not None:
            with suppress(np.linalg.LinAlgError):
                matrix = _pixel_matrix(
                    np.linalg.inv(candidate_to_source),
                    source.width,
                    source.height,
                )

    source_pixels = np.asarray(source, dtype=np.float32)
    sigma = max(source.size) / 45
    background = cv2.GaussianBlur(source_pixels, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = np.clip(source_pixels * 252 / np.maximum(background, 32), 0, 255).astype(np.uint8)
    registered = cv2.warpPerspective(
        normalized,
        matrix,
        candidate.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    registered_gray = cv2.cvtColor(registered, cv2.COLOR_RGB2GRAY)
    registered_chroma = registered.max(axis=2).astype(np.int16) - registered.min(axis=2).astype(
        np.int16
    )
    evidence = (registered_gray < 248) | ((registered_gray < 253) & (registered_chroma > 8))

    region_mask = np.zeros(registered_gray.shape, dtype=bool)
    width, height = candidate.size
    for left, top, right, bottom in regions:
        pad_x = max(0.003, (right - left) * 0.04)
        pad_y = max(0.003, (bottom - top) * 0.08)
        x1 = max(0, round((left - pad_x) * width))
        y1 = max(0, round((top - pad_y) * height))
        x2 = min(width, round((right + pad_x) * width))
        y2 = min(height, round((bottom + pad_y) * height))
        if x2 > x1 and y2 > y1:
            region_mask[y1:y2, x1:x2] = True

    result = np.asarray(candidate).copy()
    restore_mask = evidence & region_mask
    result[restore_mask] = np.minimum(result[restore_mask], registered[restore_mask])
    return Image.fromarray(result, "RGB")


def replace_with_source_evidence_regions(
    source: Image.Image,
    candidate: Image.Image,
    regions: Sequence[tuple[float, float, float, float]],
    *,
    already_aligned: bool = False,
) -> Image.Image:
    """Replace rejected regions with source evidence on a pure-white ground.

    This is intentionally different from rectangular source pasting: generated
    pixels are removed first, then only normalized authored evidence is restored.
    Paper shadows, folds, and phone-capture tone therefore cannot enter the result,
    and source text cannot double over generated text left underneath it.
    """
    source = source.convert("RGB")
    candidate = candidate.convert("RGB")
    if source.size != candidate.size or not regions:
        return candidate
    result = np.asarray(candidate).copy()
    width, height = candidate.size
    for left, top, right, bottom in regions:
        pad_x = max(0.003, (right - left) * 0.04)
        pad_y = max(0.003, (bottom - top) * 0.08)
        x1 = max(0, round((left - pad_x) * width))
        y1 = max(0, round((top - pad_y) * height))
        x2 = min(width, round((right + pad_x) * width))
        y2 = min(height, round((bottom + pad_y) * height))
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 255
    return restore_source_evidence_regions(
        source,
        Image.fromarray(result, "RGB"),
        regions,
        already_aligned=already_aligned,
    )


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
    prompt: str = REGIONAL_REPAIR_PROMPT,
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
    generated = client.generate(crop, prompt, max_edge=max_edge)
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
