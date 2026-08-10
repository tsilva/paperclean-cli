"""Conservative local fidelity gates for document-page candidates."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class DeterministicResult:
    accepted: bool
    issues: list[str]


def _registration_matrix(source: np.ndarray, candidate: np.ndarray) -> np.ndarray | None:
    max_side = 1600
    scale = min(1.0, max_side / max(source.shape))
    if scale < 1:
        size = (round(source.shape[1] * scale), round(source.shape[0] * scale))
        source_work = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
        candidate_work = cv2.resize(candidate, size, interpolation=cv2.INTER_AREA)
    else:
        source_work = source
        candidate_work = candidate
    detector = cv2.ORB_create(  # type: ignore[attr-defined]
        nfeatures=2500, fastThreshold=10
    )
    source_points, source_descriptors = detector.detectAndCompute(source_work, None)
    candidate_points, candidate_descriptors = detector.detectAndCompute(candidate_work, None)
    if source_descriptors is None or candidate_descriptors is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(candidate_descriptors, source_descriptors, k=2)
    good = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 12:
        return None
    candidate_xy = np.asarray(
        [candidate_points[item.queryIdx].pt for item in good], dtype=np.float32
    )
    source_xy = np.asarray([source_points[item.trainIdx].pt for item in good], dtype=np.float32)
    matrix, mask = cv2.findHomography(candidate_xy, source_xy, cv2.RANSAC, 4.0)
    if matrix is None or mask is None or int(mask.sum()) < 10:
        return None
    # Normalize pixel coordinates into the page-relative coordinate system.
    width = source_work.shape[1]
    height = source_work.shape[0]
    to_pixels = np.array([[width, 0, 0], [0, height, 0], [0, 0, 1]], dtype=np.float64)
    to_normalized = np.linalg.inv(to_pixels)
    return np.asarray(to_normalized @ matrix @ to_pixels, dtype=np.float64)


def _foreground_issues(
    source: Image.Image, candidate: Image.Image, candidate_to_source: np.ndarray
) -> list[str]:
    src = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cand = cv2.cvtColor(np.asarray(candidate.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if src.shape != cand.shape:
        return ["candidate_canvas_mismatch"]
    to_pixels = np.array(
        [[src.shape[1], 0, 0], [0, src.shape[0], 0], [0, 0, 1]],
        dtype=np.float64,
    )
    pixel_matrix = to_pixels @ candidate_to_source @ np.linalg.inv(to_pixels)
    cand = cv2.warpPerspective(
        cand,
        pixel_matrix,
        (src.shape[1], src.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    height, width = src.shape
    max_side = 1800
    if max(src.shape) > max_side:
        scale = max_side / max(src.shape)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        src = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        cand = cv2.resize(cand, size, interpolation=cv2.INTER_AREA)
    src_ink = cv2.adaptiveThreshold(
        src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    cand_ink = cv2.adaptiveThreshold(
        cand, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    kernel = np.ones((3, 3), np.uint8)
    src_ink = cv2.morphologyEx(src_ink, cv2.MORPH_OPEN, kernel)
    cand_ink = cv2.morphologyEx(cand_ink, cv2.MORPH_OPEN, kernel)
    content_roi = _paper_roi(src)
    src_ink = cv2.bitwise_and(src_ink, content_roi)
    cand_ink = cv2.bitwise_and(cand_ink, content_roi)
    source_pixels = int(np.count_nonzero(src_ink))
    if source_pixels < 32:
        return []
    # Regenerated glyphs have different antialiasing and stroke shapes. Compare
    # against a local support area so intentional de-skewing and pristine
    # re-typesetting do not look like wholesale foreground loss.
    support = np.ones((21, 21), np.uint8)
    candidate_support = cv2.dilate(cand_ink, support)
    source_support = cv2.dilate(src_ink, support)
    missing = cv2.bitwise_and(src_ink, cv2.bitwise_not(candidate_support))
    missing_ratio = int(np.count_nonzero(missing)) / source_pixels
    candidate_pixels = max(1, int(np.count_nonzero(cand_ink)))
    invented = cv2.bitwise_and(cand_ink, cv2.bitwise_not(source_support))
    invented_ratio = int(np.count_nonzero(invented)) / candidate_pixels
    issues: list[str] = []
    # These are intentionally permissive to exposure/background cleanup but reject
    # catastrophic content loss. Semantic review handles smaller visual changes.
    if missing_ratio > 0.45:
        issues.append("large_foreground_loss")
    if invented_ratio > 0.55:
        issues.append("large_candidate_only_foreground")
    return issues


def _paper_roi(gray: np.ndarray) -> np.ndarray:
    """Find a dominant photographed sheet, falling back to the whole canvas."""
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas_area = height * width
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < canvas_area * 0.30:
            break
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if 4 <= len(polygon) <= 8 and cv2.isContourConvex(polygon):
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            return mask
    mask = np.full_like(gray, 255, dtype=np.uint8)
    margin_x = max(1, round(width * 0.005))
    margin_y = max(1, round(height * 0.005))
    mask[:margin_y, :] = 0
    mask[-margin_y:, :] = 0
    mask[:, :margin_x] = 0
    mask[:, -margin_x:] = 0
    return mask


def _has_meaningful_foreground(gray: np.ndarray) -> bool:
    """Distinguish a genuinely blank page from one that requires registration."""
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ink = cv2.bitwise_and(ink, _paper_roi(gray))
    required = max(32, round(gray.size * 0.00002))
    return int(np.count_nonzero(ink)) >= required


def validate_candidate(
    source: Image.Image,
    candidate: Image.Image,
    *,
    min_effective_dpi: int,
    effective_dpi: float,
) -> DeterministicResult:
    issues: list[str] = []
    if effective_dpi < min_effective_dpi:
        issues.append("generated_resolution_below_minimum")
    source_gray = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate.convert("RGB")), cv2.COLOR_RGB2GRAY)
    registration = _registration_matrix(source_gray, candidate_gray)
    if registration is None:
        if _has_meaningful_foreground(source_gray):
            issues.append("page_registration_failed")
        registration = np.eye(3, dtype=np.float64)
    issues.extend(_foreground_issues(source, candidate, registration))
    return DeterministicResult(accepted=not issues, issues=issues)
