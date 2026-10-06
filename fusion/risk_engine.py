"""Risk score fusion — combines individual signals into a single decision."""

from __future__ import annotations

from typing import Any

from config.risk_weights import RISK_THRESHOLDS, RISK_WEIGHTS


def fuse(
    visual: float,
    semantic: float,
    structural: float,
    layout: float,
) -> dict[str, Any]:
    # Hard rule: mandatory semantic failure forces HIGH_RISK
    if semantic >= 1.0:
        category = "HIGH_RISK"
        decision = "SUSPICIOUS"
        final = max(
            (
                RISK_WEIGHTS["visual"] * visual
                + RISK_WEIGHTS["semantic"] * semantic
                + RISK_WEIGHTS["structural"] * structural
                + RISK_WEIGHTS["layout"] * layout
            ),
            1.0,
        )
        return {
            "risk_score": round(min(final, 1.0), 4),
            "category": category,
            "decision": decision,
        }

    final = (
        RISK_WEIGHTS["visual"] * visual
        + RISK_WEIGHTS["semantic"] * semantic
        + RISK_WEIGHTS["structural"] * structural
        + RISK_WEIGHTS["layout"] * layout
    )
    if final < RISK_THRESHOLDS["low"]:
        category = "LOW_RISK"
        decision = "APPROVE"
    elif final < RISK_THRESHOLDS["medium"]:
        category = "MEDIUM_RISK"
        decision = "REVIEW_REQUIRED"
    else:
        category = "HIGH_RISK"
        decision = "SUSPICIOUS"
    return {
        "risk_score": round(final, 4),
        "category": category,
        "decision": decision,
    }
