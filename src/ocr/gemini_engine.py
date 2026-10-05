"""Gemini OCR API wrapper — the primary OCR engine (replacing Mistral)."""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np

from src.env_config import load_project_env

# Local dev only: pulls GEMINI_API_KEY out of backend/.env. Existing
# environment variables take precedence.
load_project_env()

GEMINI_MODEL = "gemini-flash-lite-latest"

# The OCR instruction as specified
OCR_PROMPT = """Extract all visible text from this document image.

Return ONLY the OCR text.
Preserve the text exactly as it appears as much as possible.
Do not explain anything.
Do not summarize.
Do not guess missing text.
Preserve numbers, names, dates, Hindi/English text, and line structure where possible."""


def _encode_image_to_bytes(image: np.ndarray, fmt: str = "JPEG", max_dim: int = 500, quality: int = 65) -> tuple[bytes, str]:
    """Encode a numpy image array to bytes with MIME type.

    Resize to a reasonable size to avoid large payloads/timeouts.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buf = cv2.imencode(f".{fmt.lower()}", image_bgr, encode_params)
    if not success:
        raise ValueError(f"Failed to encode image as {fmt}")
    return buf.tobytes(), f"image/{fmt.lower()}"


class GeminiOCREngine:
    """Wrapper for Gemini OCR API using google-genai SDK."""

    def __init__(self, api_key: str | None = None):
        from google import genai

        # Read live so key injected after import is picked up
        self._api_key = (api_key or os.environ.get("GEMINI_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in environment or provided. "
                "Add it to backend/.env for local development, or set it as an "
                "environment variable / Secret Manager secret in deployment."
            )
        self._client = genai.Client(api_key=self._api_key)

    def run(self, image: np.ndarray) -> tuple[str, float]:
        """Run OCR on an image and return (extracted_text, average_confidence).

        Gemini does not always return detailed confidence scores in the same format
        as some OCR engines. On successful text extraction, confidence defaults to 1.0.
        """
        image_bytes, mime_type = _encode_image_to_bytes(image, "JPEG")

        from google.genai import types

        start = time.perf_counter()
        try:
            response = self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    OCR_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=60)
                ),
            )
        finally:
            elapsed = time.perf_counter() - start

        extracted = ""
        try:
            if hasattr(response, "text") and response.text:
                extracted = response.text
            elif hasattr(response, "candidates") and response.candidates:
                # Try to extract from candidates if .text is empty
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts") and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    extracted += part.text
                extracted = extracted.strip()
        except Exception:
            extracted = ""

        extracted = extracted.strip() if extracted else ""
        confidence = 1.0 if extracted else 0.0

        return extracted, confidence

    def close(self) -> None:
        # google-genai client doesn't require explicit close in most cases
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
