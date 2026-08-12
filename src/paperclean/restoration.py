"""Pristine reconstruction helpers that preserve hard-to-read source content."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
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


def _remove_punch_holes(pixels: np.ndarray) -> np.ndarray:
    """Erase dark circular binder holes only within the outer side margins."""
    result = pixels.copy()
    for center_x, center_y, radius, padding, touches_authored_ink in _punch_hole_candidates(
        result
    ):
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
            if not contained or (
                int(nearby_label) != hole_label and not centered_hole_evidence
            ):
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
                    touches_authored_ink(
                        int(center_x), int(center_y), radius, erase_padding
                    ),
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
            np.hypot(center_x - existing_x, center_y - existing_y)
            <= max(radius, existing_radius)
            for existing_x, existing_y, existing_radius, _padding, _touches in candidates
        ):
            continue
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        annulus = (squared_distance > radius**2) & (
            squared_distance <= (radius * 1.45) ** 2
        )
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
                touches_authored_ink(
                    int(center_x), int(center_y), radius, erase_padding
                ),
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
            np.hypot(center_x - existing_x, center_y - existing_y)
            <= max(radius, existing_radius)
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
                touches_authored_ink(
                    int(center_x), int(center_y), int(radius), erase_padding
                ),
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
    for center_x, center_y, radius, padding, touches_authored_ink in _punch_hole_candidates(
        pixels
    ):
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
        density = float(
            np.count_nonzero(dark[y : y + component_height, x : x + component_width])
        ) / box_area
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
    shaded = (gray < background_level - 15).astype(np.uint8)
    shade_kernel = np.ones(
        (max(5, round(height * 0.01)), max(5, round(width * 0.01))),
        np.uint8,
    )
    broad_shading = cv2.morphologyEx(shaded, cv2.MORPH_OPEN, shade_kernel)
    shade_count, shade_labels, shade_stats, _shade_centroids = (
        cv2.connectedComponentsWithStats(broad_shading, connectivity=8)
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
        component = (shade_labels == label).astype(np.uint8)
        component_y, component_x = np.where(component > 0)
        if component_x.size < 3:
            continue
        hull = cv2.convexHull(np.column_stack((component_x, component_y)).astype(np.int32))
        component.fill(0)
        cv2.fillConvexPoly(component, hull, 255)
        dilation = max(3, round(min(width, height) * 0.002))
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
    return Image.fromarray(cleaned, "RGB")


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


def restore_source_evidence_regions(
    source: Image.Image,
    candidate: Image.Image,
    regions: Sequence[tuple[float, float, float, float]],
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
    candidate_to_source = _registration_matrix(source_gray, candidate_gray)
    matrix = np.eye(3, dtype=np.float64)
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
    normalized = np.clip(source_pixels * 252 / np.maximum(background, 32), 0, 255).astype(
        np.uint8
    )
    registered = cv2.warpPerspective(
        normalized,
        matrix,
        candidate.size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    registered_gray = cv2.cvtColor(registered, cv2.COLOR_RGB2GRAY)
    registered_chroma = (
        registered.max(axis=2).astype(np.int16) - registered.min(axis=2).astype(np.int16)
    )
    evidence = (registered_gray < 248) | (
        (registered_gray < 253) & (registered_chroma > 8)
    )

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
