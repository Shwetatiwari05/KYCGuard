"""Mistral OCR API wrapper — the primary OCR engine, with EasyOCR as fallback."""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from src.env_config import load_project_env

# Local dev only: pulls MISTRAL_API_KEY out of backend/.env. Existing
# environment variables (Cloud Run env vars / Secret Manager) take precedence,
# and in a deployed image no .env file exists so this is a no-op.
load_project_env()

MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_MODEL = "mistral-ocr-latest"

# Markdown furniture Mistral emits for layout (table pipes, bold/italic markers,
# headings, image/link syntax). Left in place it glues itself to header words
# ("**भारत**", "|INCOME|TAX|") and defeats the validators' word-level matching.
_MD_NOISE_RE = re.compile(r"[*_`#|!\[\]]")


def _flatten_markdown(markdown: str) -> str:
    """Flatten a markdown page into plain, single-line text for fuzzy matching."""
    return " ".join(_MD_NOISE_RE.sub(" ", markdown.replace("\n", " ")).split())


def _encode_image_to_base64(image: np.ndarray, fmt: str = "JPEG") -> str:
    """Encode a numpy image array to base64 data URI."""
    success, buf = cv2.imencode(f".{fmt.lower()}", image)
    if not success:
        raise ValueError(f"Failed to encode image as {fmt}")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


class MistralOCREngine:
    """Wrapper for Mistral OCR API."""

    def __init__(self, api_key: str | None = None):
        import httpx
        # Read live (not at import time) so a key exported after import, or one
        # injected by the platform, is picked up.
        self._api_key = (api_key or os.environ.get("MISTRAL_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError(
                "MISTRAL_API_KEY not set in environment or provided. "
                "Add it to backend/.env for local development, or set it as an "
                "environment variable / Secret Manager secret in deployment."
            )
        # Kept short so a slow/hanging request can't stall the pipeline; on
        # timeout the caller falls back to EasyOCR.
        self._client = httpx.Client(
            timeout=httpx.Timeout(5.0, connect=3.0)
        )

    def run(self, image: np.ndarray) -> tuple[str, float]:
        """Run OCR on an image and return (extracted_text, average_confidence).

        Note: Mistral OCR does not return per-word confidence scores in the free tier,
        so confidence is estimated from the API response structure if available,
        otherwise defaults to 1.0 (assumed reliable if the call succeeds).
        """
        if image.ndim == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        data_uri = _encode_image_to_base64(image_bgr, "JPEG")

        payload = {
            "model": MISTRAL_MODEL,
            "document": {
                "type": "image_url",
                "image_url": data_uri,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        start = time.perf_counter()
        response = self._client.post(MISTRAL_OCR_URL, json=payload, headers=headers)
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            raise RuntimeError(
                f"Mistral OCR API error {response.status_code}: {response.text}"
            )

        result = response.json()
        pages = result.get("pages", [])
        texts = []
        for page in pages:
            md = page.get("markdown", "")
            if md:
                flat = _flatten_markdown(md)
                if flat:
                    texts.append(flat)

        extracted = " ".join(texts)

        confidence = 1.0
        cs = result.get("confidence_scores")
        if cs:
            confidence = cs.get("average_page_confidence_score", 1.0)

        return extracted, confidence

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
