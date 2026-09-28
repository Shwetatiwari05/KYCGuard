"""Layout/template validation — checks that OCR text lands near expected regions."""

from __future__ import annotations

from typing import Any

from src.config import REGIONS as AADHAAR_REGIONS
from src.pan_config import PAN_REGIONS


TOLERANCE = 0.20


def _rel_to_px(region: tuple[float, float, float, float], w: int, h: int) -> tuple[int, int, int, int]:
    return (
        int(region[0] * w),
        int(region[1] * h),
        int(region[2] * w),
        int(region[3] * h),
    )


def _overlaps(bbox: dict, region_px: tuple[int, int, int, int], tol: float) -> bool:
    rx1, ry1, rx2, ry2 = region_px
    rw = rx2 - rx1
    rh = ry2 - ry1
    tx1 = int(rx1 - rw * tol)
    ty1 = int(ry1 - rh * tol)
    tx2 = int(rx2 + rw * tol)
    ty2 = int(ry2 + rh * tol)

    bx1, by1, bx2, by2 = bbox["bbox"]
    if bx2 < tx1 or bx1 > tx2:
        return False
    if by2 < ty1 or by1 > ty2:
        return False
    return True


def _find_text_for_region(ocr_results: list[dict], region_px: tuple[int, int, int, int], tol: float) -> str | None:
    candidates = []
    for item in ocr_results:
        if _overlaps(item, region_px, tol):
            candidates.append(item["text"])
    if not candidates:
        return None
    return " ".join(candidates)


def validate_layout(
    ocr_results: list[dict],
    width: int,
    height: int,
    doc_type: str,
) -> tuple[float, list[str]]:
    failed: list[str] = []
    if doc_type == "AADHAAR":
        region_map = AADHAAR_REGIONS
        fields = ["name", "dob", "gender", "aadhaar_num"]
        field_labels = {
            "name": "Name",
            "dob": "Date of Birth",
            "gender": "Gender",
            "aadhaar_num": "Aadhaar number",
        }
    else:
        region_map = PAN_REGIONS
        fields = ["pan_num", "name", "father_name", "dob"]
        field_labels = {
            "pan_num": "PAN number",
            "name": "Name",
            "father_name": "Father's name",
            "dob": "Date of Birth",
        }

    missing = 0
    total = len(fields)
    for field in fields:
        if field not in region_map:
            continue
        region_px = _rel_to_px(region_map[field], width, height)
        text = _find_text_for_region(ocr_results, region_px, TOLERANCE)
        if text is None:
            missing += 1
            failed.append(f"{field_labels.get(field, field)} not found in expected region")

    risk = missing / total if total > 0 else 0.0
    return risk, failed
