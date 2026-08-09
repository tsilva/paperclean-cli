"""Conservative deterministic fidelity gates for document-page candidates."""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from paperclean.errors import ConfigurationError
from paperclean.imaging import encode_png
from paperclean.util import private_workdir, private_write

MIN_TESSERACT_VERSION = (5, 5)


@dataclass(frozen=True, slots=True)
class OcrToken:
    text: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DeterministicResult:
    accepted: bool
    issues: list[str]


def _version_tuple(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def check_tesseract(language: str) -> None:
    executable = shutil.which("tesseract")
    if executable is None:
        raise ConfigurationError("Tesseract 5.5 or newer is required on PATH")
    version = subprocess.run(
        [executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    first_line = version.stdout.splitlines()[0] if version.stdout else ""
    if version.returncode != 0 or _version_tuple(first_line) < MIN_TESSERACT_VERSION:
        raise ConfigurationError("Tesseract 5.5 or newer is required")
    languages = subprocess.run(
        [executable, "--list-langs"],
        check=False,
        capture_output=True,
        text=True,
    )
    installed = set(languages.stdout.splitlines()[1:])
    missing = [item for item in language.split("+") if item not in installed]
    if languages.returncode != 0 or missing:
        raise ConfigurationError(f"missing Tesseract language data: {', '.join(missing)}")


def ocr_tokens(image: Image.Image, language: str) -> list[OcrToken]:
    with private_workdir() as directory:
        source = directory / "page.png"
        private_write(source, encode_png(image))
        result = subprocess.run(
            ["tesseract", str(source), "stdout", "-l", language, "tsv"],
            check=False,
            capture_output=True,
        )
    if result.returncode != 0:
        raise ConfigurationError("Tesseract failed while checking a page")
    rows = csv.DictReader(
        io.StringIO(result.stdout.decode("utf-8", errors="replace")), delimiter="\t"
    )
    tokens: list[OcrToken] = []
    for row in rows:
        text = unicodedata.normalize("NFC", row.get("text", "").strip())
        try:
            confidence = float(row.get("conf", "-1"))
            left = int(row.get("left", "0"))
            top = int(row.get("top", "0"))
            width = int(row.get("width", "0"))
            height = int(row.get("height", "0"))
        except (TypeError, ValueError):
            continue
        if text and confidence >= 0 and width > 0 and height > 0:
            tokens.append(
                OcrToken(
                    text=text,
                    confidence=confidence,
                    box=(
                        left / image.width,
                        top / image.height,
                        (left + width) / image.width,
                        (top + height) / image.height,
                    ),
                )
            )
    return tokens


def _token_issues(source: list[OcrToken], candidate: list[OcrToken]) -> list[str]:
    return _registered_token_issues(source, candidate, np.eye(3, dtype=np.float64))


def _registered_token_issues(
    source: list[OcrToken], candidate: list[OcrToken], candidate_to_source: np.ndarray
) -> list[str]:
    expected = [token for token in source if token.confidence >= 80]
    issues: list[str] = []
    cursor = 0
    for token in expected:
        match: OcrToken | None = None
        match_index = cursor
        while match_index < len(candidate):
            if candidate[match_index].text == token.text:
                match = candidate[match_index]
                break
            match_index += 1
        if match is None:
            issues.append("missing_or_changed_high_confidence_text")
            break
        source_cx = (token.box[0] + token.box[2]) / 2
        source_cy = (token.box[1] + token.box[3]) / 2
        match_cx = (match.box[0] + match.box[2]) / 2
        match_cy = (match.box[1] + match.box[3]) / 2
        point = np.array([match_cx, match_cy, 1.0], dtype=np.float64)
        transformed = candidate_to_source @ point
        if abs(transformed[2]) < 1e-9:
            issues.append("page_registration_failed")
            break
        match_cx = float(transformed[0] / transformed[2])
        match_cy = float(transformed[1] / transformed[2])
        if abs(source_cx - match_cx) > 0.12 or abs(source_cy - match_cy) > 0.12:
            issues.append("high_confidence_text_moved")
            break
        cursor = match_index + 1
    return issues


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
    # Normalize pixel coordinates into the OCR token coordinate system.
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
    intersection = int(np.count_nonzero(cv2.bitwise_and(src_ink, cand_ink)))
    missing_ratio = 1 - intersection / source_pixels
    candidate_pixels = max(1, int(np.count_nonzero(cand_ink)))
    invented_ratio = 1 - intersection / candidate_pixels
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


def validate_candidate(
    source: Image.Image,
    candidate: Image.Image,
    *,
    language: str,
    min_effective_dpi: int,
    effective_dpi: float,
) -> DeterministicResult:
    issues: list[str] = []
    if effective_dpi < min_effective_dpi:
        issues.append("generated_resolution_below_minimum")
    source_tokens = ocr_tokens(source, language)
    candidate_tokens = ocr_tokens(candidate, language)
    source_gray = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2GRAY)
    candidate_gray = cv2.cvtColor(np.asarray(candidate.convert("RGB")), cv2.COLOR_RGB2GRAY)
    registration = _registration_matrix(source_gray, candidate_gray)
    if registration is None:
        # Featureless pages do not need registration; pages carrying OCR do.
        if any(token.confidence >= 80 for token in source_tokens):
            issues.append("page_registration_failed")
        registration = np.eye(3, dtype=np.float64)
    issues.extend(_registered_token_issues(source_tokens, candidate_tokens, registration))
    issues.extend(_foreground_issues(source, candidate, registration))
    return DeterministicResult(accepted=not issues, issues=issues)
