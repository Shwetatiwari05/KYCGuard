from __future__ import annotations

import os
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from PIL import Image
from typing_extensions import TypedDict

from config.risk_weights import OCR_CONFIDENCE_WARNING_THRESHOLD, RISK_WEIGHTS
from fusion.risk_engine import fuse
from training.config import CHECKPOINT_DIR
from training.model import DualBranchForgeryDetector
from src.ocr.ocr_engine import LOW_QUALITY_WARNING, OCREngine
from src.ocr.mistral_engine import MistralOCREngine
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

# OCR engines (lazy init to avoid slow import-time hangs)
_ocr_reader: OCREngine | None = None
_mistral_reader: MistralOCREngine | None = None
_engine_init_lock = threading.Lock()


def _get_ocr() -> OCREngine:
    global _ocr_reader
    if _ocr_reader is None:
        with _engine_init_lock:
            if _ocr_reader is None:
                _ocr_reader = OCREngine()
    return _ocr_reader


def _get_mistral() -> MistralOCREngine:
    global _mistral_reader
    if _mistral_reader is None:
        with _engine_init_lock:
            if _mistral_reader is None:
                _mistral_reader = MistralOCREngine()
    return _mistral_reader


# ── Timing instrumentation (parallel-validation benchmark) ─────────────────
# Records when run_ocr finishes so fuse_risk can log the OCR→fusion span,
# and stamps node starts to prove that parallel branches overlap in time.
_timing_lock = threading.Lock()
_ocr_done_at: float | None = None
_fuse_calls = 0
_node_marks: list[tuple[float, str]] = []


def _mark(node: str) -> None:
    with _timing_lock:
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


def _run_mistral(arr: np.ndarray) -> tuple[str, float]:
    """Call the primary Mistral OCR engine on a request.

    Returns (text, confidence) on success, or ("", 0.0) on any failure — missing
    MISTRAL_API_KEY, timeout, rate limit, network/API error or an unusable
    response — so the caller can fall through to EasyOCR instead of raising.
    """
    if not os.environ.get("MISTRAL_API_KEY"):
        return "", 0.0
    try:
        text, confidence = _get_mistral().run(arr)
    except Exception:
        return "", 0.0
    return text, confidence


def run_ocr(state: GraphState) -> GraphState:
    _mark("ocr")
    img = state["image"]
    arr = np.array(img)

    # Layout validation scores word bounding boxes, and only EasyOCR emits them,
    # so run it in the background while Mistral (the primary text engine) answers.
    # Wall-clock cost stays near max(mistral, easyocr) instead of their sum.
    geometry: list[tuple[list[dict], float, str | None] | None] = [None]

    def _easyocr_geometry() -> None:
        try:
            geometry[0] = _get_ocr().run(arr)
        except Exception:
            geometry[0] = None

    worker = threading.Thread(target=_easyocr_geometry, daemon=True)
    worker.start()

    mistral_text, mistral_conf = _run_mistral(arr)

    worker.join()

    easy = geometry[0]
    # None => the geometry pass failed; [] => EasyOCR ran and read nothing.
    easy_results = easy[0] if easy is not None else None
    easy_conf, easy_warning = (easy[1], easy[2]) if easy is not None else (0.0, None)

    if mistral_text.strip():
        extracted_text = normalize(mistral_text)
        ocr_source = "mistral"
        confidence = mistral_conf
        warning = (
            LOW_QUALITY_WARNING
            if confidence < OCR_CONFIDENCE_WARNING_THRESHOLD
            else None
        )
    else:
        extracted_text = " ".join(r["text"] for r in easy_results or [])
        ocr_source = "easyocr_fallback"
        confidence = easy_conf
        warning = easy_warning

    with _timing_lock:
        global _ocr_done_at
        _ocr_done_at = time.perf_counter()

    return {
        "ocr_results": easy_results,
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
    _mark("semantic")
    score, checks = _run_semantic(state)
    return {"semantic_risk_score": score, "semantic_checks": checks}


def run_structural_validation(state: GraphState) -> GraphState:
    _mark("structural")
    score, checks = _run_structural(state)
    return {"structural_risk_score": score, "structural_checks": checks}


def run_layout_validation(state: GraphState) -> GraphState:
    _mark("layout")
    score, checks = _run_layout(state)
    return {"layout_risk_score": score, "layout_checks": checks}


def fuse_risk(state: GraphState) -> GraphState:
    global _fuse_calls
    with _timing_lock:
        _fuse_calls += 1
        span = (time.perf_counter() - _ocr_done_at) * 1000 if _ocr_done_at is not None else None
        request_marks = _node_marks[:]
        _node_marks.clear()
    if span is not None:
        print(f"[timing] OCR→fusion segment = {span:.0f} ms (fuse_risk runs so far: {_fuse_calls})")
    if request_marks:
        t0 = request_marks[0][0]
        cols = ", ".join(f"{name}@{(t - t0) * 1000:+.0f}ms" for t, name in request_marks)
        print(f"[timing]   node starts: {cols}")
    result = fuse(
        visual=state.get("visual_risk_score") or 0.0,
        semantic=state.get("semantic_risk_score") or 0.0,
        structural=state.get("structural_risk_score") or 0.0,
        layout=state.get("layout_risk_score") or 0.0,
    )
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
