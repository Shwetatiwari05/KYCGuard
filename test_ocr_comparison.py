#!/usr/bin/env python3
"""OCR comparison script: EasyOCR vs Mistral OCR side-by-side.

Usage:
    python test_ocr_comparison.py [image_paths...]

If no paths are given, uses built-in synthetic samples from output/ and output_pan/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure repo root is on path
sys.path.insert(0, str(__file__).rsplit("/", 1)[0] if "/" in str(__file__) else ".")

from src.ocr.ocr_engine import OCREngine as EasyOCREngine
from src.ocr.mistral_engine import MistralOCREngine

DEFAULT_IMAGES = [
    "output/real/real_orig_100front_scaled_up.jpg",
    "output_pan/real/pan_real_0001.jpg",
]


def compare(image_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    print(f"\n{'='*70}")
    print(f"IMAGE: {image_path}  ({img.size[0]}x{img.size[1]})")
    print(f"{'='*70}")

    easy_engine = EasyOCREngine()
    t0 = time.perf_counter()
    easy_results, easy_conf, easy_warn = easy_engine.run(arr)
    easy_ms = (time.perf_counter() - t0) * 1000

    easy_text = " ".join(r["text"] for r in easy_results)
    print(f"\n[EasyOCR]  ({easy_ms:.0f} ms, avg_conf={easy_conf:.3f})")
    print(f"  Text: {easy_text[:300]}")
    if easy_warn:
        print(f"  Warning: {easy_warn}")

    try:
        mistral = MistralOCREngine()
        t1 = time.perf_counter()
        mistral_text, mistral_conf = mistral.run(arr)
        mistral_ms = (time.perf_counter() - t1) * 1000
        print(f"\n[Mistral]  ({mistral_ms:.0f} ms, conf={mistral_conf:.3f})")
        print(f"  Text: {mistral_text[:500]}")
    except Exception as e:
        print(f"\n[Mistral]  ERROR: {e}")
        mistral_ms = None
        mistral_text = ""
        mistral_conf = 0.0

    print(f"\n{'─'*70}")
    print("SIDE-BY-SIDE:")
    print(f"  EasyOCR : {easy_text[:200]}")
    print(f"  Mistral : {mistral_text[:200] if mistral_text else '(failed)'}")
    if mistral_ms is not None:
        print(f"  Latency : EasyOCR={easy_ms:.0f}ms  Mistral={mistral_ms:.0f}ms  "
              f"({'Mistral slower' if mistral_ms and mistral_ms > easy_ms * 2 else 'comparable'})")
    print(f"{'─'*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EasyOCR vs Mistral OCR")
    parser.add_argument("images", nargs="*", help="Image paths to test")
    args = parser.parse_args()

    paths = args.images if args.images else DEFAULT_IMAGES
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        print(f"Missing images: {missing}")
        sys.exit(1)

    for p in paths:
        compare(p)


if __name__ == "__main__":
    main()
