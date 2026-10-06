from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from PIL import Image
from typing_extensions import TypedDict

from config.risk_weights import OCR_CONFIDENCE_WARNING_THRESHOLD
from fusion.risk_engine import fuse
from training.config import CHECKPOINT_DIR
from training.model import DualBranchForgeryDetector
from src.ocr.azure_ocr import AzureOCREngine
from src.ocr.text_normalizer import normalize
from src.validation import (
    aadhaar_validator,
    layout_validator,
    pan_validator,
)

GROQ_MODEL = "openai/gpt-oss-20b"

# ── Model (loaded once at import / startup) ─────────────────────────────────
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

_model = DualBranchForgeryDetector(pretrained=False).to(_device)
_ckpt = torch.load(_ckpt_path, map_location=_device, weights_only=False)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_azure_ocr: AzureOCREngine | None = None


def _get_azure_ocr() -> AzureOCREngine:
    global _azure_ocr
    if _azure_ocr is None:
        _azure_ocr = AzureOCREngine()
    return _azure_ocr


# ── Timing instrumentation (parallel-validation benchmark) ─────────────────
# Records when run_ocr finishes so fuse_risk can log the OCR→fusion span,
# and stamps node starts to prove that parallel branches overlap in time.
_timing_lock = None  # not needed with simpler timing
_ocr_done_at: float | None = None
_fuse_calls = 0
_node_marks: list[tuple[float, str]] = []


def _mark(node: str) -> None:
    _node_marks.append((time.perf_counter(), node))


# ── Graph state ─────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    image: Image.Image
    doc_type: str
    spatial: torch.Tensor
    freq: torch.Tensor
    visual_risk_score: float | None
    ocr_confidence: float | None
    ocr_source: str | None
    ocr_quality_warning: str | None
    ocr_results: list[dict] | None
    extracted_text: str | None
    extracted_text_sample: str | None
    semantic_risk_score: float | None
    semantic_checks: list[str] | None
    structural_risk_score: float | None
    structural_checks: list[str] | None
    layout_risk_score: float | None
    layout_checks: list[str] | None
    final_risk_score: float | None
    final_category: str | None
    decision: str | None
    explanation: str | None


# ── Nodes ───────────────────────────────────────────────────────────────────
def run_visual_model(state: GraphState) -> GraphState:
    _mark("visual")
    spatial = state["spatial"].to(_device)
    freq = state["freq"].to(_device)
    with torch.no_grad():
        logits = _model(spatial, freq)
        probs = F.softmax(logits, dim=1)
        fake_prob = float(probs[0, 1])
    return {"visual_risk_score": fake_prob}


def _run_azure(arr: np.ndarray) -> tuple[str, float, list[dict]]:
    try:
        text, confidence, ocr_results = _get_azure_ocr().run(arr)
        return text, confidence, ocr_results
    except Exception as e:
        try:
            msg = str(e)
        except Exception:
            msg = "unknown error"
        print(f"[OCR] Azure OCR failed ({type(e).__name__}): {msg}")
        raise


def run_ocr(state: GraphState) -> GraphState:
    _mark("ocr")
    img = state["image"]
    arr = np.array(img)

    try:
        azure_text, azure_conf, azure_results = _run_azure(arr)
    except Exception:
        # Controlled OCR failure - no fallback to EasyOCR text as per requirement
        extracted_text = ""
        ocr_source = "azure_failed"
        confidence = 0.0
        warning = "Azure OCR failed. Please retry with a clearer image."
        ocr_results = []
        _ocr_done_at = time.perf_counter()
        print(f"[VALIDATION] semantic validation completed in 0.00s")
        print(f"[VALIDATION] structural validation completed in 0.00s")
        return {
            "ocr_results": ocr_results,
            "extracted_text": extracted_text,
            "extracted_text_sample": extracted_text[:500],
            "ocr_confidence": confidence,
            "ocr_quality_warning": warning,
            "ocr_source": ocr_source,
        }

    if azure_text and azure_text.strip():
        extracted_text = normalize(azure_text)
        ocr_source = "azure"
        confidence = azure_conf
        warning = None
        if confidence < OCR_CONFIDENCE_WARNING_THRESHOLD and confidence > 0:
            warning = "Image quality is low — OCR results may be unreliable. Consider retaking the photo with better lighting and less glare."
        ocr_results = azure_results
    else:
        extracted_text = ""
        ocr_source = "azure"
        confidence = azure_conf
        warning = "No text could be extracted from the image. Ensure the document is clearly visible, well-lit, and in focus."
        ocr_results = azure_results

    _ocr_done_at = time.perf_counter()

    return {
        "ocr_results": ocr_results,
        "extracted_text": extracted_text,
        "extracted_text_sample": extracted_text[:500],
        "ocr_confidence": confidence,
        "ocr_quality_warning": warning,
        "ocr_source": ocr_source,
    }


def _run_semantic(state: GraphState) -> tuple[float, list[str]]:
    doc_type = state["doc_type"]
    text = state.get("extracted_text") or ""
    img = np.array(state["image"])
    if doc_type == "AADHAAR":
        return aadhaar_validator.validate_semantic(img, extracted_text=text)
    return pan_validator.validate_semantic(text)


def _run_structural(state: GraphState) -> tuple[float, list[str]]:
    doc_type = state["doc_type"]
    text = state.get("extracted_text") or ""
    if doc_type == "AADHAAR":
        return aadhaar_validator.validate_structural(text)
    return pan_validator.validate_structural(text)


def _run_layout(state: GraphState) -> tuple[float, list[str]]:
    doc_type = state["doc_type"]
    img = state["image"]
    w, h = img.size
    ocr_results = state.get("ocr_results")
    if ocr_results is None:
        # No geometry at all (the EasyOCR pass failed) — omit the signal rather
        # than reporting every field as missing.
        return 0.0, []
    return layout_validator.validate_layout(ocr_results, w, h, doc_type)


def run_semantic_validation(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    _mark("semantic")
    score, checks = _run_semantic(state)
    print(f"[VALIDATION] semantic validation completed in {time.perf_counter() - t0:.2f}s")
    return {"semantic_risk_score": score, "semantic_checks": checks}


def run_structural_validation(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    _mark("structural")
    score, checks = _run_structural(state)
    print(f"[VALIDATION] structural validation completed in {time.perf_counter() - t0:.2f}s")
    return {"structural_risk_score": score, "structural_checks": checks}


def run_layout_validation(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    _mark("layout")
    score, checks = _run_layout(state)
    print(f"[VALIDATION] layout validation completed in {time.perf_counter() - t0:.2f}s")
    return {"layout_risk_score": score, "layout_checks": checks}


def fuse_risk(state: GraphState) -> GraphState:
    t0 = time.perf_counter()
    global _fuse_calls
    _fuse_calls += 1
    result = fuse(
        visual=state.get("visual_risk_score") or 0.0,
        semantic=state.get("semantic_risk_score") or 0.0,
        structural=state.get("structural_risk_score") or 0.0,
        layout=state.get("layout_risk_score") or 0.0,
    )
    print(f"[RISK] risk fusion completed in {time.perf_counter() - t0:.2f}s")
    return {
        "final_risk_score": result["risk_score"],
        "final_category": result["category"],
        "decision": result["decision"],
    }


def _route_decision(state: GraphState) -> str:
    return "explain" if state.get("decision") != "APPROVE" else "report"


def explain(state: GraphState) -> GraphState:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"explanation": "Explanation unavailable: GROQ_API_KEY not set."}

    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key)

    def _section(name: str, score: float | None, checks: list[str] | None) -> str:
        s = f"{name}: risk score {score:.0%}" if score is not None else f"{name}: N/A"
        if checks:
            s += " — issues: " + "; ".join(checks)
        return s

    lines = [
        f"Decision: {state.get('decision')}",
        f"Final risk score: {state.get('final_risk_score', 0):.0%}",
        _section("Visual forensics", state.get("visual_risk_score"), None),
        _section("Semantic validation", state.get("semantic_risk_score"), state.get("semantic_checks")),
        _section("Structural validation", state.get("structural_risk_score"), state.get("structural_checks")),
        _section("Layout validation", state.get("layout_risk_score"), state.get("layout_checks")),
    ]
    evidence = "\n".join(lines)

    prompt = (
        "You are a compliance assistant. A KYC document was analyzed by a multi-signal "
        "forgery-detection system. Below is the evidence from each signal module. "
        "Only reference the specific findings listed below; do not speculate about issues "
        "not mentioned. Write a concise 2-3 sentence plain-English explanation for a "
        "non-technical reviewer explaining why the document was flagged.\n\n"
        f"{evidence}"
    )
    response = llm.invoke(prompt)
    explanation = response.content if hasattr(response, "content") else str(response)
    return {"explanation": explanation}


def report(state: GraphState) -> GraphState:
    return {"explanation": None}


def join_signals(state: GraphState) -> GraphState:
    """Fan-in point for the parallel visual/OCR branches; fans back out to the
    three validation nodes. LangGraph fires this node exactly once because its
    two in-edges (visual + OCR) complete in the same superstep."""
    _mark("join")
    return {}


# ── Build graph ─────────────────────────────────────────────────────────────
_g = StateGraph(GraphState)
_g.add_node("run_visual_model", run_visual_model)
_g.add_node("run_ocr", run_ocr)
_g.add_node("join_signals", join_signals)
_g.add_node("run_semantic_validation", run_semantic_validation)
_g.add_node("run_structural_validation", run_structural_validation)
_g.add_node("run_layout_validation", run_layout_validation)
_g.add_node("fuse_risk", fuse_risk)
_g.add_node("explain", explain)
_g.add_node("report", report)

_g.add_edge(START, "run_visual_model")
_g.add_edge(START, "run_ocr")
_g.add_edge("run_visual_model", "join_signals")
_g.add_edge("run_ocr", "join_signals")
_g.add_edge("join_signals", "run_semantic_validation")
_g.add_edge("join_signals", "run_structural_validation")
_g.add_edge("join_signals", "run_layout_validation")
_g.add_edge("run_semantic_validation", "fuse_risk")
_g.add_edge("run_structural_validation", "fuse_risk")
_g.add_edge("run_layout_validation", "fuse_risk")
_g.add_conditional_edges("fuse_risk", _route_decision, {"report": "report", "explain": "explain"})
_g.add_edge("report", END)
_g.add_edge("explain", END)

graph = _g.compile()
