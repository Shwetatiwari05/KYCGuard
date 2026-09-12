# KYCGuard AI — Multi-Signal KYC Document Risk Assessment

> A research pipeline that flags potentially forged Indian KYC documents (Aadhaar & PAN) by combining **visual forensics, OCR + semantic validation, structural validation,** and **layout consistency** into a single fused risk score.

> **⚠️ Research prototype — not a production KYC verification system.** This software does not verify identity and is not a substitute for official verification with government records. "Format-valid" here means the document matches expected syntax and layout heuristics — it does **not** mean the document is authentic or government-verified.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [The Four Risk Signals](#the-four-risk-signals)
- [Risk Fusion & Decisions](#risk-fusion--decisions)
- [Model Details](#model-details)
- [Evaluation](#evaluation)
- [Dataset Generation](#dataset-generation)
- [Setup & Installation](#setup--installation)
- [Running the Backend](#running-the-backend)
- [Running the Frontend](#running-the-frontend)
- [API Reference](#api-reference)
- [Known Limitations](#known-limitations)
- [Project Structure](#project-structure)

---

## What It Does

KYCGuard analyses a scanned or phone-captured image of an Indian KYC document and produces a **risk assessment** instead of a naive pass/fail.

For each document it runs **four independent signals**, each tuned to catch a different class of forgery:

1. **Visual forensics** — a dual-branch CNN looks for pixel-level manipulation, compression artefacts, blur, noise, splicing and affine warps in both the spatial and the frequency (DCT) domain.
2. **OCR + semantic validation** — extracts the document text and fuzzy-matches it against the official headers and expected content for the document type, catching swapped or misspelled content.
3. **Structural validation** — checks the format of key identifiers (12-digit Aadhaar number, PAN `AAAAA9999A` pattern).
4. **Layout consistency** — verifies that required fields are present *where they are supposed to be*, using calibrated template regions.

The four signal scores are fused with fixed weights into a single `final_risk_score`, which maps to a category (`LOW_RISK` / `MEDIUM_RISK` / `HIGH_RISK`) and a decision (`APPROVE` / `REVIEW_REQUIRED` / `SUSPICIOUS`). When a document is *not* approved, an LLM (Groq) writes a short plain-English explanation of *why*, grounded strictly in the evidence produced by the four signals.

The system is exposed through a FastAPI service (`POST /predict`) and a two-page vanilla-JS frontend: a landing page and a live demo tool with real webcam capture.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Frontend (vanilla JS)                            │
│   Landing page (index.html)          Demo tool (demo.html)                  │
│                                        ├─ Upload (file input)               │
│                                        └─ Live webcam capture (getUserMedia │
│                                           → canvas → JPEG blob → File)      │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │  POST /predict
                                           │  multipart: file + doc_type (AADHAAR|PAN)
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI backend (backend/main.py)                   │
│                                POST /predict                                │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│            LangGraph pipeline (backend/graph.py)                            │
│                                                                             │
│   START                                                                     │
│     └─▶ run_visual_model      dual-branch CNN (EfficientNet-B0 + DCT-CNN)   │
│              │                → visual_risk_score (fake probability)        │
│              ▼                                                              │
│     └─▶ run_ocr               EasyOCR (en+hi, CPU, preprocess: deskew,      │
│              │                CLAHE, denoise)                               │
│              │                ─ if avg confidence < 0.75 ─▶ Mistral OCR API  │
│              │                   fallback (ocr_source = "mistral_fallback") │
│              ▼                                                              │
│     └─▶ run_semantic_validation   fuzzy-match official headers vs OCR text  │
│              │                    (Aadhaar: "GOVERNMENT OF INDIA",          │
│              │                    "भारत सरकार"; PAN: "INCOME TAX DEPARTMENT",│
│              │                    "GOVT. OF INDIA")                        │
│              ▼                                                              │
│     └─▶ run_structural_validation  Aadhaar 12-digit / PAN regex check       │
│              ▼                                                              │
│     └─▶ run_layout_validation      OCR bboxes vs calibrated template regions│
│              ▼                                                              │
│     └─▶ fuse_risk                    weighted sum → category + decision     │
│              │                                                              │
│              ├─ decision == APPROVE ──▶ report   (no explanation)          │
│              │                                                              │
│              └─ decision != APPROVE ─▶ explain  (Groq LLM, conditional on   │
│                                          GROQ_API_KEY set)                  │
│              │                                                              │
│              ▼                                                              │
│     END  ──▶ JSON response                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

The graph nodes run **sequentially** (visual → OCR → semantic → structural → layout → fuse), then a conditional edge routes to the LLM explanation or straight to the report.

---

## Tech Stack

Pulled from [`requirements.txt`](requirements.txt) and the actual imports:

| Area | Technologies |
|------|--------------|
| **Dataset generation** | OpenCV, Pillow, NumPy, Pandas, tqdm, DeepFace (gender/age face analysis, via `tf-keras`), requests |
| **Training** | PyTorch, torchvision, timm (EfficientNet-B0), SciPy (DCT), scikit-learn (metrics/splits), matplotlib, seaborn |
| **Backend / orchestration** | FastAPI, uvicorn, python-multipart, LangGraph, langchain-groq, python-dotenv |
| **OCR / validation** | EasyOCR (en + hi, CPU), rapidfuzz (fuzzy matching), httpx (Mistral OCR API fallback) |
| **Frontend** | Vanilla HTML/CSS/JS + Lucide icons + Google Fonts (CDN) — no build step |

Optional runtime keys (loaded from `.env` / `backend/.env`):

- `GROQ_API_KEY` — enables the LLM explanation layer (`openai/gpt-oss-20b`). Without it, `explanation` is returned as `"Explanation unavailable: GROQ_API_KEY not set."`
- `MISTRAL_API_KEY` — optional; enables the Mistral OCR fallback when EasyOCR confidence is low. OCR works fine without it.

---

## The Four Risk Signals

### 1. Visual forensics (`run_visual_model`)
- A dual-branch CNN classifies the document as real/fake and reports the **probability of "fake"** as `visual_risk_score`.
- **Spatial branch:** EfficientNet-B0 (ImageNet-pretrained, pooled to 1280-d features).
- **Frequency branch:** 3-layer DCT-CNN (32→64→128 channels) operating on log-scaled, normalised 2-D DCT coefficients — designed to surface compression/artefact fingerprints that can be subtle in pixel space.
- Both branches feed a learned fusion head.
- Regions the model leans on can be inspected with Grad-CAM (`training/grad_cam.py`).

### 2. OCR + semantic validation (`run_ocr` → `run_semantic_validation`)
- EasyOCR (`["en", "hi"]`, CPU) runs on a preprocessed image (deskew → CLAHE → bilateral denoise). Results are text-normalised and each line is emitted with a confidence and bounding box.
- **Hybrid fallback:** if the average OCR confidence is below `0.75`, the text is re-extracted with the **Mistral OCR API** (`mistral-ocr-latest`). On success the text replaces the EasyOCR output, the confidence is boosted, and `ocr_source` is set to `"mistral_fallback"`; otherwise EasyOCR output stands (`ocr_source = "easyocr"`).
- A quality warning is attached when average confidence falls below `0.5` or nothing is read.
- Semantic validation fuzzy-matches (rapidfuzz) the expected official headers against the extracted text:
  - **Aadhaar:** `GOVERNMENT OF INDIA`, `भारत सरकार` (the Hindi header is also OCR'd on a 4× upscaled header-region crop; known EasyOCR Devanagari/English erratums — `भारत→भरत`, `INDIYA→INDIA` — are normalised before scoring).
  - **PAN:** `INCOME TAX DEPARTMENT`, `GOVT. OF INDIA`.
  - Any expected string scoring below 88% fuzzy match is recorded as a failed check; risk = `1.0` if any failed, else `0.0`.

### 3. Structural validation (`run_structural_validation`)
- **Aadhaar:** a 12-digit number must be present (whitespace/hyphens stripped before matching).
- **PAN:** the number must match `[A-Z]{5}[0-9]{4}[A-Z]`.
- Missing/invalid identifier ⇒ risk `1.0`; else `0.0`.

### 4. Layout consistency (`run_layout_validation`)
- The OCR bounding boxes are checked for overlap with **calibrated, relative-coordinate template regions** (tolerance ±20% per region):
  - **Aadhaar:** name, DOB, gender, Aadhaar number (`REGIONS` in `src/config.py`).
  - **PAN:** PAN number, applicant name, father's name, DOB (`PAN_REGIONS` in `src/pan_config.py`).
- Risk = fraction of expected fields not found in their regions.

The per-signal risk thresholds are more lenient than the fuse thresholds because the final score is the weighted combination:

---

## Risk Fusion & Decisions

Fusion (`fusion/risk_engine.py`), driven by fixed weights in `config/risk_weights.py`:

| Signal | Weight |
|--------|--------|
| Visual forensics | 0.45 |
| Semantic validation | 0.30 |
| Structural validation | 0.15 |
| Layout consistency | 0.10 |

```
final_risk_score = 0.45·visual + 0.30·semantic + 0.15·structural + 0.10·layout
```

| final_risk_score | Category | Decision |
|------------------|----------|----------|
| < 0.30 | `LOW_RISK` | `APPROVE` |
| < 0.60 | `MEDIUM_RISK` | `REVIEW_REQUIRED` |
| ≥ 0.60 | `HIGH_RISK` | `SUSPICIOUS` |

Only non-`APPROVE` documents trigger the LLM explanation node.

---

## Model Details

`training/model.py` — **`DualBranchForgeryDetector`**

| Component | Details |
|-----------|---------|
| Spatial branch | EfficientNet-B0 (timm, ImageNet-pretrained, `num_classes=0`, global avg pool) → 1280-d |
| Frequency branch | Conv2D 32→64→128 → BN/ReLU/MaxPool → adaptive avg pool → FC → 128-d |
| Fusion head | `concat(1280, 128) = 1408` → Linear 256 → Linear 64 → Linear 2 (logits) |
| Loss | Focal loss (α=0.25, γ=2.0) |
| Optimizer | AdamW, weight decay 1e-4, gradient clipping 1.0 |

**Two-phase training** (`training/train.py`):
- **Phase 1 (backbone frozen):** 5 epochs @ LR 1e-3, trains frequency branch + fusion head only.
- **Phase 2 (full fine-tune):** up to 30 epochs, differential LRs — backbone 1e-5, frequency 2.5e-4, fusion 5e-4 — with cosine annealing warm restarts and early stopping (patience 10).

**Data hygiene:** `training/dataset.py` uses an **identity-aware split** (`GroupShuffleSplit` keyed on `face_file`) — every document rendered from the same face goes into the same split, so the model cannot memorise faces instead of learning forgery patterns. The final reported run had zero train↔val/train↔test face leakage. Near-real-time inference uses MPS on Apple Silicon, CPU otherwise.

---

## Evaluation

Reported metrics from the current run — `results/test_metrics.json`, evaluated on the **held-out identity-aware test set (145 samples)**, no face leakage:

| Metric | Score |
|--------|-------|
| **Accuracy** | **92.41%** |
| **Precision** | 96.43% |
| **Recall** | 85.71% |
| **F1 score** | 90.76% |
| **AUC-ROC** | **96.15%** |

Artifacts are written to `results/` (`confusion_matrix.png`, `roc_curve.png`, `training_curves.png`, `training_history.json`).

> Precision is markedly higher than recall — when KYCGuard flags a document it is very likely to be fake, but it will still miss some forgeries. Treat the visual signal as a screening layer, not proof.

---

## Dataset Generation

KYCGuard's CNN is trained on **procedurally generated synthetic documents**, not on collected real forgeries.

**Aadhaar pipeline** (`src/pipeline.py`):
1. Copy the 90 original real Aadhaar scans → `output/real/` (label 0).
2. Extract clean, blank-card templates from real scans (`src/template_extractor.py`; region calibration via `calibrate.py`).
3. Analyse candidate face photos with DeepFace (gender + age) → cache.
4. Generate **200 real-synthetic cards** (label 0) where face gender ↔ name, face age ↔ DOB — all consistent.
5. Generate **200 fake cards** (label 1), each combining **2–3 tampering categories**, sampled with:

| Category | Prob. | Example |
|----------|-------|---------|
| `semantic_gender`/`age`/`aadhaar` | 0.50 | gender swap, age mismatch, malformed Aadhaar |
| `partial_editing` | 0.40 | edit a single field with a font/shift/char-spacing penalty |
| `face_tampering` | 0.35 | face brightness / compositing mismatch |
| `text_tampering` | 0.45 | font variation, character shift, blurred text |
| `image_quality` | 0.60 | JPEG compression (q 15–45), Gaussian blur/noise, colour |
| `structural` | 0.25 | affine warp (~±3°), global shifts |
| `border_crop` | 0.20 | cropped edges, copy-paste border artifacts |

6. Write `output/dataset.csv` → **490 samples** (290 real = 90 originals + 200 synthetic; 200 fake).

**PAN pipeline** (`src/pan_pipeline.py`): a single calibrated PAN template (`src/clean_pan_sample.png`, `calibrate_pan.py`) is used to render **600 samples** (`output_pan/pan_dataset.csv`: 300 real / 300 fake) with the same tamper-category scheme and valid-format PAN numbers.

**Training input** (`training/dataset.py`) merges both CSVs (≈1,090 samples) and augments the spatial view at train time (rotation, horizontal flip, brightness, contrast, Gaussian blur).

---

## Setup & Installation

```bash
# 1. Create a virtual environment (Python 3.10+ recommended)
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets (backend/ loading order: .env, then backend/.env)
cp backend/.env.example backend/.env
#    edit backend/.env:
#    GROQ_API_KEY=your_groq_key_here        → enables LLM explanations
#    MISTRAL_API_KEY=your_key_here          → optional, enables OCR fallback
```

**Optional — regenerate the dataset:**

```bash
# Aadhaar cards (needs data/aadhaar/real/ scans + data/faces/ photos)
python -m src.pipeline

# PAN cards (uses the bundled sample template)
python -m src.pan_pipeline
```

Face analysis hits DeepFace, which downloads model weights + faces on demand; `--dry-run` analyses faces without rendering cards.

**Optional — (re)train the model:**

```bash
python -m training.train          # two-phase: frozen backbone, then fine-tune
python -m training.evaluate       # writes metrics + plots to results/
python -m training.grad_cam --image path/to/document.jpg   # explainability
```

The backend already ships a trained checkpoint (`checkpoints/best_model.pt`), so training is not required to run the service. Device is picked automatically — MPS on Apple Silicon, CUDA if available in `training.train`, CPU otherwise.

---

## Running the Backend

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

A reload is deliberately avoided (`--reload` is not used): the module loads the model checkpoint at import time and lazily spins up the EasyOCR reader on the first request, and a reload watchdog would re-run that heavy initialisation on every file change. Start it once and leave it.

The service listens for `/predict` (see [API Reference](#api-reference)) and exposes interactive docs at `http://127.0.0.1:8000/docs`.

---

## Running the Frontend

The frontend is static files — serve the `frontend/` folder over HTTP:

```bash
cd frontend
python -m http.server 5500
```

- Landing page: `http://localhost:5500/index.html`
- Demo tool:  `http://localhost:5500/demo.html` (try it with Aadhaar/PAN samples, upload, or the live webcam capture)

The demo posts the captured/uploaded image to `http://localhost:8000/predict` (`frontend/demo.js`), so keep the backend and the static server running together. It renders the four signal scores, the fused decision, and — when present — the LLM explanation.

> The initial model + OCR warm-up can take a few seconds on the first request.

---

## API Reference

### `POST /predict`

Multipart/form-data:

| Field | Type | Description |
|-------|------|-------------|
| `file` | image/* | The document image (JPEG/PNG) |
| `doc_type` | string | `AADHAAR` (default) or `PAN` |

Example:

```bash
curl -F "file=@aadhaar.jpg" -F "doc_type=AADHAAR" http://127.0.0.1:8000/predict
```

### Response schema

```jsonc
{
  "document_type": "AADHAAR",
  "visual_forensics": {
    "risk_score": 0.7314,          // fake probability from the dual-branch CNN
    "status": "HIGH_RISK"          // LOW_RISK / MEDIUM_RISK / HIGH_RISK (score <0.3 / <0.6 / else)
  },
  "ocr": {
    "confidence": 0.8231,          // average confidence of the OCR run
    "source": "easyocr",           // "easyocr" | "mistral_fallback"
    "quality_warning": null,       // set when avg conf < 0.5 or nothing read
    "extracted_text_sample": "..." // first ~500 chars of extracted text
  },
  "semantic_validation": {
    "risk_score": 1.0,             // 0.0 or 1.0
    "failed_checks": ["Official English header mismatch: expected 'GOVERNMENT OF INDIA' (fuzzy score 61%)"]
  },
  "structural_validation": {
    "risk_score": 0.0,
    "failed_checks": []
  },
  "layout_validation": {
    "risk_score": 0.25,            // fraction of expected fields missing from their region
    "failed_checks": ["Name not found in expected region"]
  },
  "final_decision": {
    "risk_score": 0.62,            // fused: 0.45·visual + 0.30·semantic + 0.15·structural + 0.10·layout
    "category": "HIGH_RISK",       // LOW_RISK / MEDIUM_RISK / HIGH_RISK
    "decision": "SUSPICIOUS"       // APPROVE / REVIEW_REQUIRED / SUSPICIOUS
  },
  "explanation": "The document was flagged because ..."  // null if approved, or if GROQ_API_KEY unset
}
```

Notes:
- `explanation` is only produced for non-`APPROVE` decisions **and** when `GROQ_API_KEY` is set; otherwise it is `null` or the "unavailable" message.
- Errors: `400` for non-image uploads, empty files, or an invalid `doc_type`.

---

## Known Limitations

These are honest, code-verified constraints — not feature claims:

1. **Synthetic-to-real domain gap.** The visual CNN was trained on procedurally rendered forgeries (composited cards + JPEG/blur/noise/warp/tamper categories). Heavily degraded, adversarially edited, or retaken (photo-of-screen / re-printed) documents can fall outside the training distribution, which is reflected in the recall of 85.7% — the model misses some fakes.
2. **Single-template layout calibration.** Layout validation checks against one set of hand-calibrated relative regions per document type (Aadhaar `REGIONS`, PAN `PAN_REGIONS`, ±20% tolerance). Newer or alternate-format editions of the same card may not match those regions, causing false layout flags.
3. **OCR accuracy on severely degraded images.** EasyOCR runs on CPU and its confidence drops sharply under glare, blur, dark ink, or low resolution. That is mitigated by the Mistral OCR fallback (which needs `MISTRAL_API_KEY` + network) and by quality warnings, but semantic and structural signals are only as good as the extracted text.
4. **Known OCR erratums.** EasyOCR mis-reads of Devanagari/English headers (e.g. `भारत→भरत`, `INDIYA→INDIA`) are patched by specific normalisation rules, not handled generically.
5. **LLM explanations are evidence-grounded but require an API key** and only run for flagged documents. They cite only the findings from the four signals; they never perform official verification.
6. **Format-valid ≠ authentic.** Structural/layout checks verify syntax and placement, not provenance. A well-formed document can still be inauthentic, and validation here is not a substitute for government verification.
7. **Not a production system.** No document storage, no audit trail, no rate limiting, no concurrency tuning; the model checkpoint is loaded into memory per worker and OCR is cold-started on first request.

---

## Project Structure

```
./  (repo root)
├── backend/
│   ├── main.py            # FastAPI app — POST /predict
│   ├── graph.py           # LangGraph orchestration + OCR fallback + LLM explanation
│   └── .env.example       # GROQ_API_KEY (+ optional MISTRAL_API_KEY)
├── config/
│   └── risk_weights.py    # fusion weights + risk thresholds
├── fusion/
│   └── risk_engine.py     # weighted risk fusion → category + decision
├── src/
│   ├── pipeline.py        # Aadhaar dataset generation orchestrator
│   ├── pan_pipeline.py    # PAN dataset generation orchestrator
│   ├── config.py          # Aadhaar template REGIONS + fake-category probabilities
│   ├── pan_config.py      # PAN template PAN_REGIONS + fake config
│   ├── card_composer.py   # Aadhaar card image compositor
│   ├── pan_card_composer.py  # PAN card image compositor
│   ├── template_extractor.py # blank-template extraction from real scans
│   ├── augmentor.py       # post-render forgery augmentations
│   ├── face_analyzer.py   # DeepFace gender/age analysis + cache
│   ├── data_utils.py      # names, DOB, Aadhaar/PAN number generators
│   ├── ocr/
│   │   ├── ocr_engine.py      # EasyOCR wrapper (en+hi, CPU)
│   │   ├── mistral_engine.py  # Mistral OCR API fallback
│   │   ├── preprocessing.py   # deskew / CLAHE / denoise
│   │   └── text_normalizer.py # NFC + whitespace normalisation
│   └── validation/
│       ├── aadhaar_validator.py  # semantic + structural checks (Aadhaar)
│       ├── pan_validator.py      # semantic + structural checks (PAN)
│       └── layout_validator.py   # bbox-vs-region layout checks
├── training/
│   ├── model.py           # DualBranchForgeryDetector + FocalLoss
│   ├── config.py          # hyperparameters / paths
│   ├── train.py           # two-phase training loop
│   ├── dataset.py         # identity-aware split + dual-view dataset
│   ├── dct_utils.py       # log-scaled 2-D DCT
│   ├── evaluate.py        # metrics + plots → results/
│   └── grad_cam.py        # Grad-CAM visualisation
├── frontend/
│   ├── index.html         # landing page (signals overview)
│   ├── demo.html          # demo tool (upload + webcam)
│   ├── styles.css         # shared styling
│   ├── landing.js         # hero animation, typewriter, scroll effects
│   └── demo.js            # capture/upload → /predict → result rendering
├── checkpoints/           # best_model.pt / final_model.pt
├── results/               # test_metrics.json, confusion/ROC/curves PNGs
├── output/                # Aadhaar dataset (dataset.csv + images)
├── output_pan/            # PAN dataset (pan_dataset.csv + images)
├── data/                  # real scans, faces, templates, fonts
├── calibrate.py           # Aadhaar region calibration tool
├── calibrate_pan.py       # PAN region calibration tool
└── requirements.txt
```

---

## Licence

Not specified. Research prototype — see the disclaimer at the top of this document before using it for anything other than experimentation.