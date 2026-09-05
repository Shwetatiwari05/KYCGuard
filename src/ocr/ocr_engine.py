"""EasyOCR wrapper — runs once per image and returns structured output."""

from __future__ import annotations

import cv2
import numpy as np

from config.risk_weights import OCR_PREPROCESSING_ENABLED, OCR_CONFIDENCE_WARNING_THRESHOLD
from .preprocessing import preprocess
from .text_normalizer import normalize_ocr_result


class OCREngine:
    def __init__(self):
        import easyocr
        self._reader = easyocr.Reader(["en", "hi"], gpu=False, verbose=False)

    def run(self, image: np.ndarray) -> tuple[list[dict], float, str | None]:
        """Run OCR and return (results, avg_confidence, quality_warning)."""
        if OCR_PREPROCESSING_ENABLED:
            ocr_input = preprocess(image)
        else:
            ocr_input = image

        raw = self._reader.readtext(ocr_input, detail=1, paragraph=False)
        cleaned = normalize_ocr_result(raw)

        results = []
        for bbox, text, conf in cleaned:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            results.append({
                "text": text,
                "confidence": conf,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })

        avg_conf = (
            sum(r["confidence"] for r in results) / len(results) if results else 0.0
        )

        warning = None
        if not results:
            warning = (
                "No text could be extracted from the image. "
                "Ensure the document is clearly visible, well-lit, and in focus."
            )
        elif avg_conf < OCR_CONFIDENCE_WARNING_THRESHOLD:
            warning = (
                "Image quality is low — OCR results may be unreliable. "
                "Consider retaking the photo with better lighting and less glare."
            )

        return results, avg_conf, warning

    def run_on_region(
        self, image: np.ndarray, region: tuple[float, float, float, float], upscale: int = 4
    ) -> tuple[list[dict], float]:
        """Crop to a relative region, upscale, run OCR, return (results, avg_confidence)."""
        h, w = image.shape[:2]
        x1 = int(region[0] * w)
        y1 = int(region[1] * h)
        x2 = int(region[2] * w)
        y2 = int(region[3] * h)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return [], 0.0

        up_w = max(1, crop.shape[1] * upscale)
        up_h = max(1, crop.shape[0] * upscale)
        upscaled = cv2.resize(crop, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

        results, avg_conf, _ = self.run(upscaled)
        return results, avg_conf
