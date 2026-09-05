"""Text normalization utilities for OCR output."""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = " ".join(text.split())
    return text


def normalize_ocr_result(result: list[tuple]) -> list[tuple]:
    out = []
    for bbox, text, conf in result:
        out.append((bbox, normalize(text), float(conf)))
    return out
