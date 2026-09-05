"""OCR-specific image preprocessing: grayscale, deskew, CLAHE, denoise."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply best-effort preprocessing for OCR on phone-captured KYC images.

    Args:
        image: BGR or RGB uint8 numpy array.

    Returns:
        Grayscale uint8 numpy array, preprocessed for OCR.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = _deskew(gray)
    gray = _clahe(gray)
    gray = _denoise(gray)

    return gray


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Correct rotation/skew using the minimum-area bounding box of text contours."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 20:
        return gray

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    if abs(angle) < 0.5:
        return gray

    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, 5, 50, 50)
