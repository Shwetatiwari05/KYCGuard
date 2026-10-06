"""Azure AI Vision OCR wrapper - primary OCR engine."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.env_config import load_project_env

load_project_env()


def _encode_image_to_bytes(image: np.ndarray, fmt: str = "JPEG", max_dim: int = 2000, quality: int = 85) -> bytes:
    """Encode numpy image to bytes."""
    import cv2

    if image.ndim == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img = image

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buf = cv2.imencode(f".{fmt.lower()}", img, params)
    if not success:
        raise ValueError(f"Failed to encode image as {fmt}")
    return buf.tobytes()


class AzureOCREngine:
    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None):
        self.endpoint = (endpoint or os.environ.get("AZURE_VISION_ENDPOINT") or "").rstrip("/")
        self.key = (key or os.environ.get("AZURE_VISION_KEY") or "").strip()
        if not self.endpoint or not self.key:
            raise ValueError("AZURE_VISION_ENDPOINT and AZURE_VISION_KEY must be set")

    def _post_with_retry(self, image_bytes: bytes, timeout: int = 20) -> Dict[str, Any]:
        import requests

        url = f"{self.endpoint}/computervision/imageanalysis:analyze?api-version=2024-02-01&features=read"
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream",
        }

        # Try once, retry once on transient errors
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
                if resp.status_code == 200:
                    return resp.json()
                # For some transient codes, retry
                if attempt == 0 and resp.status_code in (429, 500, 503):
                    time.sleep(0.5)
                    continue
                raise RuntimeError(f"Azure Vision API error {resp.status_code}: {resp.text[:200]}")
            except (requests.RequestException,) as e:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise
        raise RuntimeError("Azure OCR request failed after retries")

    def run(self, image: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Run Azure OCR. Returns (full_text, avg_conf, ocr_results in layout format)."""
        t0 = time.perf_counter()
        print("[OCR] Azure request started")
        image_bytes = _encode_image_to_bytes(image)
        result = self._post_with_retry(image_bytes, timeout=20)
        elapsed = time.perf_counter() - t0
        print(f"[OCR] Azure request completed in {elapsed:.2f}s")

        ocr_results: List[Dict[str, Any]] = []
        texts: List[str] = []
        confs: List[float] = []

        read_result = result.get("readResult", {})
        lines = []
        if "lines" in read_result:
            lines = read_result.get("lines", []) or []
        else:
            pages = read_result.get("pages", []) or []
            if pages:
                for page in pages:
                    for line in page.get("lines", []) or []:
                        lines.append(line)
            else:
                blocks = read_result.get("blocks", []) or []
                for block in blocks:
                    for line in block.get("lines", []) or []:
                        lines.append(line)
        for line in lines:
            line_text = line.get("text", "") or ""
            if line_text:
                texts.append(line_text)
            for word in line.get("words", []) or []:
                wtext = word.get("text", "") or ""
                wconf = word.get("confidence", 1.0)
                if isinstance(wconf, (int, float)):
                    confs.append(float(wconf))
                poly = word.get("boundingPolygon", []) or word.get("polygon", []) or []
                xs, ys = [], []
                if poly and isinstance(poly[0], dict):
                    for p in poly:
                        if "x" in p and "y" in p:
                            xs.append(float(p["x"]))
                            ys.append(float(p["y"]))
                elif len(poly) >= 4:
                    for i in range(0, len(poly), 2):
                        xs.append(float(poly[i]))
                        ys.append(float(poly[i + 1]))
                if xs and ys:
                    bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    bbox = [0, 0, 0, 0]
                ocr_results.append({
                    "text": wtext,
                    "confidence": float(wconf) if isinstance(wconf, (int, float)) else 1.0,
                    "bbox": bbox,
                })

        full_text = " ".join(texts) if texts else " ".join(r["text"] for r in ocr_results)
        avg_conf = sum(confs) / len(confs) if confs else (1.0 if full_text else 0.0)

        print(f"[OCR] extracted {len(texts)} lines / {len(ocr_results)} words")

        return full_text, avg_conf, ocr_results
