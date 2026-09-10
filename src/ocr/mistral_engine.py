"""Mistral OCR API wrapper — cloud-based OCR as an alternative to EasyOCR."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_OCR_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_MODEL = "mistral-ocr-latest"


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
        self._api_key = api_key or MISTRAL_API_KEY
        if not self._api_key:
            raise ValueError("MISTRAL_API_KEY not set in environment or provided.")
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
                texts.append(md)

        extracted = "\n".join(texts)

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
