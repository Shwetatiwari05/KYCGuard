import io

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from training.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from training.dct_utils import compute_dct

from backend.graph import graph

load_dotenv()

app = FastAPI(title="FinShield — KYC Forgery Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _preprocess(file_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor, Image.Image]:
    """Replicate training/grad_cam.prepare_image on raw upload bytes."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32)

    arr = img_np / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    spatial = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(_device)

    dct_arr = compute_dct(img_np)
    freq = torch.from_numpy(dct_arr.transpose(2, 0, 1)).unsqueeze(0).to(_device)

    return spatial, freq, img


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    doc_type: str = Form("AADHAAR"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc_type = doc_type.upper().strip()
    if doc_type not in ("AADHAAR", "PAN"):
        raise HTTPException(status_code=400, detail="doc_type must be AADHAAR or PAN.")

    spatial, freq, img = _preprocess(file_bytes)
    result = graph.invoke(
        {
            "image": img,
            "doc_type": doc_type,
            "spatial": spatial,
            "freq": freq,
            "visual_risk_score": None,
            "ocr_confidence": None,
            "ocr_quality_warning": None,
            "ocr_results": None,
            "extracted_text": None,
            "extracted_text_sample": None,
            "semantic_risk_score": None,
            "semantic_checks": None,
            "structural_risk_score": None,
            "structural_checks": None,
            "layout_risk_score": None,
            "layout_checks": None,
            "final_risk_score": None,
            "final_category": None,
            "decision": None,
            "explanation": None,
        }
    )

    return {
        "document_type": doc_type,
        "visual_forensics": {
            "risk_score": round(result["visual_risk_score"], 4),
            "status": _to_status(result["visual_risk_score"]),
        },
        "ocr": {
            "confidence": round(result["ocr_confidence"], 4) if result["ocr_confidence"] is not None else None,
            "quality_warning": result.get("ocr_quality_warning"),
            "extracted_text_sample": result.get("extracted_text_sample", ""),
        },
        "semantic_validation": {
            "risk_score": round(result["semantic_risk_score"], 4) if result["semantic_risk_score"] is not None else None,
            "failed_checks": result.get("semantic_checks") or [],
        },
        "structural_validation": {
            "risk_score": round(result["structural_risk_score"], 4) if result["structural_risk_score"] is not None else None,
            "failed_checks": result.get("structural_checks") or [],
        },
        "layout_validation": {
            "risk_score": round(result["layout_risk_score"], 4) if result["layout_risk_score"] is not None else None,
            "failed_checks": result.get("layout_checks") or [],
        },
        "final_decision": {
            "risk_score": round(result["final_risk_score"], 4) if result["final_risk_score"] is not None else None,
            "category": result.get("final_category"),
            "decision": result.get("decision"),
        },
        "explanation": result.get("explanation"),
    }


def _to_status(score: float) -> str:
    if score < 0.3:
        return "LOW_RISK"
    if score < 0.6:
        return "MEDIUM_RISK"
    return "HIGH_RISK"
