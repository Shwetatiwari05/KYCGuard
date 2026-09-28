"""Aadhaar-specific semantic and structural validation."""

from __future__ import annotations

import re

import numpy as np
from rapidfuzz import fuzz

from src.config import HEADER_REGION
from src.ocr.ocr_engine import OCREngine


AADHAAR_EXPECTED_STRINGS = [
    ("GOVERNMENT OF INDIA", "Official English header"),
    ("भारत सरकार", "Official Hindi header"),
]

FUZZY_THRESHOLD = 88
AADHAAR_NUM_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")

_ocr = OCREngine()


def _extract_aadhaar_number(text: str) -> str | None:
    clean = re.sub(r"[\s\-]", "", text)
    m = AADHAAR_NUM_RE.search(clean)
    if m:
        return m.group(0)
    return None


def _validate_phrase(expected: str, text: str) -> float:
    partial = fuzz.partial_ratio(expected.lower(), text.lower())
    exp_words = expected.lower().split()
    text_words = text.lower().split()
    if not exp_words:
        return partial
    min_word_score = 100.0
    for ew in exp_words:
        best = 0.0
        for tw in text_words:
            score = fuzz.ratio(ew, tw)
            if score > best:
                best = score
                if best == 100:
                    break
        if best < min_word_score:
            min_word_score = best
    return min(partial, min_word_score)


def _header_crop_ocr(image: np.ndarray) -> str:
    """Run targeted OCR on the header region and return extracted text."""
    results, _ = _ocr.run_on_region(image, HEADER_REGION, upscale=4)
    return " ".join(r["text"] for r in results)


def _normalize_hindi_header(text: str) -> str:
    """Normalize known EasyOCR Devanagari erratum: 'भारत' → 'भरत' (missing aa matra)."""
    return text.replace("भारत", "भरत")


def _normalize_english_header(text: str) -> str:
    """Normalize known EasyOCR English header erratums on Aadhaar cards."""
    return text.replace("OE", "OF").replace("INDIYA", "INDIA")


def validate_semantic(image_or_text: str | np.ndarray, extracted_text: str = "") -> tuple[float, list[str]]:
    """Validate semantic content.

    Args:
        image_or_text: Either the raw image array (for targeted header OCR) or the full extracted text.
        extracted_text: Full extracted text from whole-image OCR.
    """
    failed: list[str] = []
    for expected, label in AADHAAR_EXPECTED_STRINGS:
        whole_score = _validate_phrase(expected, extracted_text)

        if "भारत सरकार" in expected:
            header_text = ""
            if isinstance(image_or_text, np.ndarray):
                header_text = _header_crop_ocr(image_or_text)
            crop_score = _validate_phrase(expected, header_text) if header_text else 0.0
            
            norm_expected = _normalize_hindi_header(expected)
            norm_whole = _normalize_hindi_header(extracted_text)
            norm_crop = _normalize_hindi_header(header_text) if header_text else ""
            norm_whole_score = _validate_phrase(norm_expected, norm_whole)
            norm_crop_score = _validate_phrase(norm_expected, norm_crop) if norm_crop else 0.0
            score = max(whole_score, crop_score, norm_whole_score, norm_crop_score)
        elif "GOVERNMENT OF INDIA" in expected:
            norm_expected = _normalize_english_header(expected)
            norm_whole = _normalize_english_header(extracted_text)
            norm_whole_score = _validate_phrase(norm_expected, norm_whole)
            score = max(whole_score, norm_whole_score)
        else:
            score = whole_score

        if score < FUZZY_THRESHOLD:
            failed.append(
                f"{label} mismatch: expected '{expected}' (fuzzy score {score}%)"
            )
    risk = 1.0 if failed else 0.0
    return risk, failed


def validate_structural(extracted_text: str) -> tuple[float, list[str]]:
    failed: list[str] = []
    num = _extract_aadhaar_number(extracted_text)
    if num is None:
        failed.append("Aadhaar number not found in extracted text")
        return 1.0, failed
    if len(num) != 12 or not num.isdigit():
        failed.append(f"Aadhaar number format invalid: '{num}'")
        return 1.0, failed
    return 0.0, failed
