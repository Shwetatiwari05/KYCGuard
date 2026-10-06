"""Aadhaar-specific semantic and structural validation."""

from __future__ import annotations

import re

from rapidfuzz import fuzz


AADHAAR_EXPECTED_STRINGS = [
    ("GOVERNMENT OF INDIA", "Official English header"),
    ("भारत सरकार", "Official Hindi header"),
]

FUZZY_THRESHOLD = 88
# Hard threshold for mandatory phrases to catch clear spelling errors
MANDATORY_FUZZY_THRESHOLD = 95
AADHAAR_NUM_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")


def _extract_aadhaar_number(text: str) -> str | None:
    if not text:
        return None
    clean = re.sub(r"[\s\-\.\,]", "", text)
    m = AADHAAR_NUM_RE.search(clean)
    if m:
        return m.group(0)
    # Try to extract sequences of 4 digits separated by spaces
    m = re.search(r"(\d{4})\s+(\d{4})\s+(\d{4})", text)
    if m:
        num = m.group(1) + m.group(2) + m.group(3)
        if len(num) == 12 and num.isdigit():
            return num
    # Fallback: find any 12 consecutive digits anywhere
    m = re.search(r"(\d{12})", clean)
    if m:
        return m.group(1)
    return None


def _validate_phrase(expected: str, text: str) -> float:
    # Normalize whitespace and case
    expected_norm = re.sub(r'\s+', ' ', expected.lower().strip())
    text_norm = re.sub(r'\s+', ' ', text.lower().strip())
    partial = fuzz.partial_ratio(expected_norm, text_norm)
    exp_words = expected_norm.split()
    text_words = text_norm.split()
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


def _normalize_hindi_header(text: str) -> str:
    """Normalize known OCR Devanagari erratum: 'भारत' → 'भरत' (missing aa matra)."""
    return text.replace("भारत", "भरत")


def _normalize_english_header(text: str) -> str:
    """Normalize known OCR English header erratums on Aadhaar cards."""
    return text.replace("OE", "OF").replace("INDIYA", "INDIA")


def validate_semantic(image_or_text=None, extracted_text: str = "") -> tuple[float, list[str]]:
    """Validate semantic content.

    Args:
        image_or_text: Ignored (kept for backward compatibility).
        extracted_text: Full extracted text from whole-image OCR.
    """
    if extracted_text is None:
        extracted_text = ""

    failed: list[str] = []
    for expected, label in AADHAAR_EXPECTED_STRINGS:
        whole_score = _validate_phrase(expected, extracted_text)

        if "भारत सरकार" in expected:
            norm_expected = _normalize_hindi_header(expected)
            norm_whole = _normalize_hindi_header(extracted_text)
            norm_whole_score = _validate_phrase(norm_expected, norm_whole)
            score = max(whole_score, norm_whole_score)
        elif "GOVERNMENT OF INDIA" in expected:
            norm_expected = _normalize_english_header(expected)
            norm_whole = _normalize_english_header(extracted_text)
            norm_whole_score = _validate_phrase(norm_expected, norm_whole)
            score = max(whole_score, norm_whole_score)
        else:
            score = whole_score

        # Use stricter threshold for mandatory Aadhaar phrases
        threshold = MANDATORY_FUZZY_THRESHOLD if expected in ("GOVERNMENT OF INDIA", "भारत सरकार") else FUZZY_THRESHOLD
        if score < threshold:
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
