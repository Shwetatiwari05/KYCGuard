"""PAN-specific semantic and structural validation."""

from __future__ import annotations

import re

from rapidfuzz import fuzz


PAN_EXPECTED_STRINGS = [
    ("INCOME TAX DEPARTMENT", "Official English header"),
    ("GOVT. OF INDIA", "Official government header"),
]

FUZZY_THRESHOLD = 88
PAN_NUM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Za-z0-9])")
PAN_NUM_RE_STRICT = re.compile(r"(?:^|[^A-Za-z0-9])([A-Z]{5}[0-9]{4}[A-Z])(?=[^A-Za-z0-9]|$)")


def _extract_pan_number(text: str) -> str | None:
    m = PAN_NUM_RE.search(text)
    if m:
        return m.group(0)
    clean = re.sub(r"[\s\-]", "", text)
    m = PAN_NUM_RE_STRICT.search(clean)
    if m:
        return m.group(1)
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


def _normalize_english_header(text: str) -> str:
    """Normalize known EasyOCR PAN header erratums."""
    return re.sub(r"\s+", "", text)


def validate_semantic(extracted_text: str) -> tuple[float, list[str]]:
    failed: list[str] = []
    for expected, label in PAN_EXPECTED_STRINGS:
        whole_score = _validate_phrase(expected, extracted_text)

        if "INCOME TAX DEPARTMENT" in expected:
            norm_expected = _normalize_english_header(expected)
            norm_whole = _normalize_english_header(extracted_text)
            norm_whole_score = fuzz.partial_ratio(norm_expected.lower(), norm_whole.lower())
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
    num = _extract_pan_number(text=extracted_text)
    if num is None:
        failed.append("PAN number not found in extracted text")
        return 1.0, failed
    return 0.0, failed
