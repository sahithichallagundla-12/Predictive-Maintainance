# Aegis: Industrial Predictive Maintenance Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/ChromaDB-Local%20Vector%20Store-orange?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Explainability-SHAP-blueviolet?style=for-the-badge" alt="SHAP" />
  <img src="https://img.shields.io/badge/LLM-Groq%20Cloud-f55036?style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## Executive Summary

**Aegis** is an end-to-end industrial intelligence and predictive maintenance command center built on the **AI4I 2020** dataset. It integrates six specialized binary machine learning classifiers, local Explainable AI (SHAP), an offline local Retrieval-Augmented Generation (RAG) knowledge engine using ChromaDB, and an optional two-stage Groq-powered diagnosis and strict fact-verification workflow.

The system is architected with a **privacy-first, local-first** design: sensor ingestion, feature engineering, multi-target classification, SHAP explanations, and maintenance manual retrieval all execute completely offline on your local machine.

> [!WARNING]
> **Research & Prototyping Prototype:** Aegis is evaluated on the synthetic AI4I 2020 dataset and is designed for research, bench-testing, and demonstration. Model predictions do not replace physical inspection or safety certifications. Qualified maintenance and plant engineering personnel must validate all diagnostic recommendations prior to taking operational action.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Dataset & Target Failure Modes](#dataset--target-failure-modes)
  - [Domain Feature Engineering](#domain-feature-engineering)
  - [Candidate Models & Imbalance Handling](#candidate-models--imbalance-handling)
  - [Adaptive Probability Threshold Optimization](#adaptive-probability-threshold-optimization)
  - [Model Benchmark & Evaluation Results](#model-benchmark--evaluation-results)
- [Explainability & Root Cause Analysis (SHAP)](#explainability--root-cause-analysis-shap)
- [Local RAG Knowledge Engine](#local-rag-knowledge-engine)
  - [Document Ingestion & Chunking](#document-ingestion--chunking)
  - [Dynamic Manual Hot-Swapping](#dynamic-manual-hot-swapping)
- [Two-Stage LLM Diagnosis & Verification](#two-stage-llm-diagnosis--verification)
- [Web Command Center (UI)](#web-command-center-ui)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Quick Start](#installation--quick-start)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository & Setup Environment](#1-clone-repository--setup-environment)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Train or Retrain Models (Optional)](#3-train-or-retrain-models-optional)
  - [4. Launch the Application](#4-launch-the-application)
- [REST API Specification](#rest-api-specification)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Limitations & Industrial Disclaimers](#limitations--industrial-disclaimers)
- [License](#license)

---

## System Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │           Industrial Operator / UI            │
                  │   FastAPI Web Dashboard (Dark Command Center) │
                  └───────┬───────────────────────────────▲───────┘
                          │ Telemetry Inputs              │ Visualizations & Reports
                          │ (Speed, Temp, Torque, Wear)   │ (Gauge, Cards, SHAP, AI)
                          ▼                               │
    ┌───────────────────────────────────────────────┐     │
    │         Feature Engineering Pipeline          │     │
    │  • ΔT (Process - Air Temp)                    │     │
    │  • Mechanical Power (Torque × ω)              │     │
    │  • Torque-Wear Interaction                    │     │
    └───────┬───────────────────────────────────────┘     │
            │ Cleaned & Engineered Tensor                 │
            ▼                                             │
    ┌───────────────────────────────────────────────┐     │
    │     Multi-Target Classifier Ensemble          ├─────┤
    │  • Machine Failure    • Tool Wear (TWF)       │     │ Probabilities &
    │  • Heat Dissip. (HDF) • Power (PWF)           │     │ Predictions
    │  • Overstrain (OSF)   • Random (RNF)          │     │
    └───────┬───────────────────────────────────────┘     │
            │                                             │
            ├─────────────────────────────────────────────┤
            ▼                                             │
    ┌───────────────────────────────────────────────┐     │
    │       Explainable AI Engine (SHAP)            ├─────┤ Local Feature
    │  Calculates Top-5 Attributions per Target     │     │ Attributions
    └───────┬───────────────────────────────────────┘     │
            │ Risk Drivers & Probabilities                │
            ▼                                             │
    ┌───────────────────────────────────────────────┐     │
    │       Local ChromaDB Vector Store             │     │
    │  • 100% Offline (all-MiniLM-L6-v2)            ├─────┤ Top-4 Retrieved
    │  • PDF / DOCX Technical Manual Passages       │     │ Technical Passages
    └───────┬───────────────────────────────────────┘     │
            │ Context + Predictions + Sources             │
            ▼                                             │
    ┌───────────────────────────────────────────────┐     │
    │     Two-Stage LLM Diagnosis (Groq Cloud)      │     │
    │  Stage 1: Root Cause & Action Synthesis       │     │ Verified Decision
    │  Stage 2: Strict Evidence Faithfulness Audit  ├─────┘ Support Object
    └───────────────────────────────────────────────┘
```

---

## Key Features

- **Multi-Target Failure Detection:** Concurrently analyzes incoming operating conditions against **6 independent failure modes**, accounting for co-occurring and multi-component mechanical breakdowns.
- **Explainable Predictions (SHAP):** Deconstructs black-box predictions into top-5 feature attributions with clear directional indicators (*e.g., elevated torque increases failure risk*).
- **100% Offline Core Operations:** Ingestion, model scoring, SHAP calculation, and document retrieval operate without internet access or external third-party dependencies.
- **Local RAG Integration:** Automatically ingests and indexes machinery operating manuals (PDF/DOCX) using local embeddings (`all-MiniLM-L6-v2`) and ChromaDB vector persistence.
- **Dynamic Manual Hot-Swapping:** Upload and index new technical documentation directly through the browser with automated SHA-256 deduplication and obsolete index cleanup.
- **Two-Stage Hallucination-Checked AI Diagnosis:** Employs an optional Groq LLM to synthesize maintenance recommendations, followed immediately by an automated evidence verifier that cross-references assertions against cited manual passages.
- **Modern Industrial Dashboard:** High-contrast dark telemetry console featuring an animated SVG failure gauge, dynamic threshold indicators, real-time input validation, and interactive source citations.

---

## Machine Learning Pipeline

### Dataset & Target Failure Modes

The pipeline is trained on the **AI4I 2020 Predictive Maintenance Dataset** containing **10,000 operational observations** of milling equipment. Data is split using a deterministic seed (`42`) stratified across overall machine failure:

| Split | Ratio | Samples | Objective |
|:---|:---:|:---:|:---|
| **Training Set** | 70% | 7,000 | Feature preprocessing, scaling, and candidate model fitting |
| **Validation Set** | 15% | 1,500 | Multi-algorithm comparison and optimal threshold calibration |
| **Held-Out Test Set** | 15% | 1,500 | Unbiased final performance and generalization evaluation |

Six distinct binary classifiers are trained independently, allowing the system to identify both general instability and specific mechanical root causes:
1. `Machine failure`: Overall breakdown indicator (macro failure).
2. `TWF` (Tool Wear Failure): Tool reaches critical wear thresholds and requires replacement.
3. `HDF` (Heat Dissipation Failure): Inadequate heat transfer causing critical overheating.
4. `PWF` (Power Failure): Mechanical power levels fall outside acceptable operational bounds.
5. `OSF` (Overstrain Failure): Product of tool wear and mechanical strain exceeds structural limits.
6. `RNF` (Random Failure): Uncorrelated failure mode with no standard parametric signature.

### Domain Feature Engineering

Raw input attributes are enriched with physical first-principles representations of equipment dynamics:

$$\Delta T = T_{\text{process}} - T_{\text{air}} \quad [\text{K}]$$

$$P_{\text{mech}} = \tau \times \omega = \tau \times \left(\frac{2\pi \times \text{rpm}}{60}\right) \quad [\text{W}]$$

$$\text{Interaction} = \tau \times W_{\text{tool}} \quad [\text{Nm} \cdot \text{min}]$$

```python
# Raw Features: Type (L/M/H), Air Temp [K], Process Temp [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min]
# Engineered Features:
engineered["Temperature difference [K]"] = engineered["Process temperature [K]"] - engineered["Air temperature [K]"]
engineered["Mechanical power [W]"] = engineered["Torque [Nm]"] * 2.0 * np.pi * engineered["Rotational speed [rpm]"] / 60.0
engineered["Torque-wear interaction"] = engineered["Torque [Nm]"] * engineered["Tool wear [min]"]
```

Pre-processing utilizes `sklearn.compose.ColumnTransformer`:
- Numeric columns: Median imputation + optional `StandardScaler` (active for neural network candidates).
- Categorical column (`Type`): Most-frequent imputation + `OneHotEncoder(handle_unknown='ignore')`.
- Identifiers (`UDI`, `Product ID`): Strictly excluded from model inputs.

### Candidate Models & Imbalance Handling

For every failure target, `train.py` systematically trains and compares **seven algorithms**:
1. **Random Forest** (`RandomForestClassifier`)
2. **Extra Trees** (`ExtraTreesClassifier`)
3. **XGBoost** (`XGBClassifier`)
4. **LightGBM** (`LGBMClassifier`)
5. **CatBoost** (`CatBoostClassifier`)
6. **HistGradientBoosting** (`HistGradientBoostingClassifier`)
7. **Multi-Layer Perceptron** (`MLPClassifier`)

Class imbalance is addressed dynamically using algorithmic weighting:
- Balanced class weights (`class_weight="balanced"`) for tree ensembles and HistGradientBoosting.
- Positive-class scale weights (`scale_pos_weight = N_neg / N_pos`) for XGBoost and LightGBM.
- Sample weight vector generation (`compute_sample_weight("balanced")`) where applicable.

### Adaptive Probability Threshold Optimization

Rather than adopting an arbitrary default cutoff ($p = 0.50$), each target algorithm's decision threshold is calibrated independently:
1. Candidate models output probabilities for the validation partition.
2. A discrete search sweeps 19 thresholds: $\tau \in [0.05, 0.95]$ with step $0.05$.
3. The candidate model and threshold combination maximizing the validation $F_1$-score is selected.
4. Ties are broken in favor of higher validation Recall, followed by the lower threshold.
5. Final evaluation is computed on the held-out test split using the chosen threshold.

### Model Benchmark & Evaluation Results

The table below summarizes the best-performing models selected by validation $F_1$ and their performance on the held-out test set (`artifacts/training/best_models_summary.json`):

| Failure Target | Selected Model | Chosen Cutoff | Val $F_1$ | Test Precision | Test Recall | Test $F_1$ | Test ROC-AUC | Test PR-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Machine Failure** | HistGradientBoosting | **0.65** | 0.8800 | 0.8600 | 0.8431 | **0.8515** | 0.9891 | 0.9067 |
| **Heat Dissipation (HDF)** | HistGradientBoosting | **0.05** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| **Power Failure (PWF)** | Random Forest | **0.30** | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 |
| **Overstrain (OSF)** | LightGBM | **0.05** | 1.0000 | 0.8824 | 1.0000 | **0.9375** | 1.0000 | 1.0000 |
| **Tool Wear (TWF)** | CatBoost | **0.45** | 0.2353 | 0.0000 | 0.0000 | **0.0000** | 0.9462 | 0.0517 |
| **Random Failure (RNF)** | Extra Trees | **0.25** | 0.2500 | 0.0000 | 0.0000 | **0.0000** | 0.8344 | 0.0333 |

> [!NOTE]
> **Extreme Imbalance Notice:** In synthetic benchmarks, rare failure modes like `TWF` and `RNF` have minimal positive representation in test partitions (only 2 positive test cases for RNF). While ROC-AUC remains high, precision/recall metrics reflect the challenge of detecting micro-classes in small validation samples.

---

## Explainability & Root Cause Analysis (SHAP)

For every inference request, Aegis generates real-time feature attributions:
- Uses `shap.TreeExplainer` for native tree architectures (HistGradientBoosting, LightGBM, CatBoost, RandomForest).
- Falls back automatically to model-agnostic permutation explainers (`shap.Explainer`) for non-tree estimators.
- Ranks and extracts the **top 5 most influential features** contributing to or mitigating each failure prediction.
- Each attribution provides directionality:
  - **Increases risk:** Positive SHAP contribution pushing probability toward trigger threshold.
  - **Decreases risk:** Negative SHAP contribution keeping probability in safe zones.

---

## Local RAG Knowledge Engine

Aegis includes an embedded knowledge retrieval system designed to operate without third-party vector API services or cloud infrastructure.

```
 Technical Manuals (.pdf / .docx)
               │
               ▼
   Page & Table Text Extraction
               │
               ▼
   Recursive Sliding Window Chunking (180 words, 35-word overlap)
               │
               ▼
   ChromaDB Embedding Pipeline (sentence-transformers: all-MiniLM-L6-v2)
               │
               ▼
   Persistent Vector Store (artifacts/chroma/chroma.sqlite3)
               │
               ▼
   Cosine Similarity Retrieval (Top-4 Semantic Passages)
```

### Document Ingestion & Chunking

1. **Extraction:**
   - **PDF:** Extracts selectable text per page using `pypdf.PdfReader` with encryption checks.
   - **DOCX:** Iterates across paragraphs and structured tabular cells via `python-docx`.
2. **Chunking Strategy:**
   - Chunks text into sliding windows of **180 words** with **35 words of overlap** to preserve procedural context across section boundaries.
3. **Local Vector Storage:**
   - Stored in a persistent local Chroma collection named `predictive_maintenance_manual` using cosine distance metrics (`hnsw:space: cosine`).
4. **Retrieval Queries:**
   - Queries are synthesized dynamically combining predicted failure modes, probability values, and leading SHAP feature drivers to locate targeted maintenance actions.

### Dynamic Manual Hot-Swapping

Operators can upload updated standard operating procedures (SOPs) or equipment-specific service manuals directly via the dashboard:
- Supports files up to **20 MB** (`.pdf`, `.docx`).
- Pre-validates file integrity and text extractability prior to updating the vector index.
- Computes **SHA-256 digests** to avoid duplicate indexing of previously registered manuals.
- Atomic cutover: Old vector chunks are purged only after the new manual is verified and fully indexed.

---

## Two-Stage LLM Diagnosis & Verification

To eliminate hallucinations common in generative maintenance recommendations, Aegis employs a **two-phase verification pipeline** powered by Groq:

```
Telemetry + Predictions + SHAP + Top-4 Retrieved Manual Passages
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │     Stage 1: Diagnostic Report Synthesis      │
         │  • Root-cause analysis                       │
         │  • Concrete maintenance actions              │
         │  • Mandatory source citations [S1], [S2]     │
         └──────────────────────┬───────────────────────┘
                                │ Draft Diagnosis Object
                                ▼
         ┌──────────────────────────────────────────────┐
         │     Stage 2: Strict Faithfulness Auditor     │
         │  • Audits assertions against manual passages │
         │  • Identifies ungrounded or invented claims  │
         │  • Flags evidence gaps                       │
         │  • Status: verified | partially | unverified │
         └──────────────────────┬───────────────────────┘
                                │
                                ▼
         Final Verified Maintenance Advisory Presented to Operator
```

1. **Stage 1 (Synthesis):** The model receives sensor readings, probabilities, SHAP drivers, and retrieved manual excerpts. It generates a diagnostic report with mandatory citation markers (`[S1]`, `[S2]`).
2. **Stage 2 (Faithfulness Audit):** A second, strictly instructed verifier agent receives only the draft report and the retrieved excerpts. It audits every statement to verify whether the source text directly supports the claim, labeling the advisory as `verified`, `partially_verified`, or `not_verified`.

---

## Web Command Center (UI)

The frontend is served directly by FastAPI as an interactive single-page application built with modern vanilla JavaScript and custom industrial CSS:

- **Telemetry Control Center:** Form inputs with range validation for Machine Class (`L`, `M`, `H`), temperatures, rotational speed, torque, and tool wear.
- **Dynamic Probability Gauge:** Interactive SVG dial displaying machine failure probability and calibrated decision threshold.
- **Failure Mode Cards:** Visual progress bars displaying each individual failure mode's probability versus its calibrated threshold.
- **SHAP Breakdown Panel:** Instant visualization of dominant sensor factors driving predictions up or down.
- **Knowledge Base Manager:** Displays the active manual source name, chunk count, and provides a drag-and-drop replacement modal.
- **Audited Diagnosis Viewer:** Renders markdown diagnostic advice, flagged evidence gaps, and expandable source excerpts with relevance scoring.

---

## Project Directory Structure

```
.
├── ai4i2020.csv                          # Primary dataset (10,000 observations)
├── app.py                                # FastAPI backend application & API routes
├── train.py                              # Machine learning training & evaluation pipeline
├── requirements.txt                      # Project Python dependencies
├── .env.example                          # Environment variable template
├── .gitignore                            # Git exclusion rules
├── predictive maintanance manual.pdf     # Default seed technical maintenance manual
├── static/                               # Web dashboard assets
│   ├── index.html                        # Application interface markup
│   ├── styles.css                        # Industrial theme stylesheet
│   └── app.js                            # Telemetry handlers, SVG gauge & API calls
├── artifacts/                            # Model weights, reports, and vector storage
│   ├── chroma/                           # Persistent ChromaDB database files
│   └── training/                         # Model training outputs
│       ├── best_models_summary.json      # Summary metrics and chosen thresholds
│       ├── model_validation_comparison.csv# Evaluation results across all candidate models
│       ├── run_metadata.json             # Dataset split information and training metadata
│       ├── models/                       # Serialized Joblib model bundles
│       │   ├── best_machine_failure.joblib
│       │   ├── best_twf.joblib
│       │   ├── best_hdf.joblib
│       │   ├── best_pwf.joblib
│       │   ├── best_osf.joblib
│       │   └── best_rnf.joblib
│       └── reports/                      # Per-target test predictions and JSON reports
└── data/                                 # Runtime document uploads
    └── manuals/                          # User-uploaded manual files (ignored by git)
```

---

## Installation & Quick Start

### Prerequisites

- **Python:** Version 3.9, 3.10, or 3.11 installed on your system.
- **C/C++ Build Tools:** Standard toolchains required for building wheel distributions (if pre-built wheels are unavailable).
- **Groq API Key:** (Optional) Required only if you want to use LLM diagnostic reports. Core ML inference, SHAP explanations, and RAG retrieval operate without an API key.

---

### 1. Clone Repository & Setup Environment

#### On Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### On Linux / macOS (Bash / Zsh)

```bash
# Clone the repository
git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Configure Environment Variables

Create your `.env` configuration file from the provided template:

```bash
# Copy template
cp .env.example .env
```

Edit `.env` to supply your configuration:

```ini
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your_actual_groq_api_key_here
```

> [!TIP]
> If you do not have a Groq API key, you can still run the entire application. The **Predict & explain** feature and local vector manual retrieval will work completely offline. Only the **Full AI diagnosis** button requires a key.

---

### 3. Train or Retrain Models (Optional)

Pre-trained model bundles are included in `artifacts/training/models/`. If you want to train from scratch on new data or verify the pipeline:

```bash
# Run training pipeline with defaults (seed 42, writes to artifacts/training/)
python train.py
```

Advanced training CLI options:

```bash
python train.py --data "path/to/ai4i2020.csv" --output-dir "artifacts/training" --seed 42
```

---

### 4. Launch the Application

Start the FastAPI application using Uvicorn:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
- **Interactive UI Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc Docs:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Service Health Endpoint:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## REST API Specification

### 1. Health Status
`GET /api/health`

Returns operational readiness of model bundles, active vector collection, and LLM configuration.

**Response Example:**
```json
{
  "status": "ready",
  "models": {
    "ready": true,
    "loaded": true,
    "available_targets": ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
  },
  "documents": {
    "count": 1,
    "items": [{ "filename": "predictive maintanance manual.pdf", "type": "pdf" }],
    "index": {
      "ready": true,
      "provider": "chroma",
      "collection": "predictive_maintenance_manual",
      "embedding_model": "all-MiniLM-L6-v2",
      "chunks": 42
    }
  },
  "llm": {
    "provider": "groq",
    "model": "openai/gpt-oss-120b",
    "configured": true
  }
}
```

---

### 2. Predict & Explain (Local)
`POST /api/predict`

Executes multi-target binary inference and computes SHAP feature attributions.

**Request Payload:**
```json
{
  "machine_type": "M",
  "air_temperature_k": 303.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1400.0,
  "torque_nm": 50.0,
  "tool_wear_min": 220.0
}
```

**Response Example:**
```json
{
  "risk_level": "high",
  "machine_failure_probability": 0.892,
  "predictions": [
    {
      "target": "Machine failure",
      "model": "HistGradientBoosting",
      "probability": 0.892,
      "threshold": 0.65,
      "predicted": true
    },
    {
      "target": "OSF",
      "model": "LightGBM",
      "probability": 0.941,
      "threshold": 0.05,
      "predicted": true
    }
  ],
  "shap_explanations": {
    "Machine failure": [
      {
        "feature": "Torque-wear interaction",
        "shap_value": 0.412,
        "direction": "increases"
      },
      {
        "feature": "Tool wear [min]",
        "shap_value": 0.285,
        "direction": "increases"
      }
    ]
  }
}
```

---

### 3. Full AI Diagnosis & Verified Actions
`POST /api/diagnose`

Runs local prediction, queries the local Chroma vector store, synthesizes actions via Groq, and performs strict faithfulness verification.

**Request Payload:** Same as `/api/predict`.

**Response Example:**
```json
{
  "risk_level": "high",
  "machine_failure_probability": 0.892,
  "diagnosis": {
    "diagnosis_markdown": "Elevated torque under extended tool wear indicates imminent overstrain failure [S1].",
    "recommended_actions": [
      "Halt milling operations immediately and disengage spindle drive [S1].",
      "Inspect cutter insert for micro-chipping and replace tool assembly [S2]."
    ],
    "evidence_gaps": ["Coolant flow rate telemetry is unavailable in current stream."]
  },
  "faithfulness": {
    "status": "verified",
    "supported_claims": [
      "Elevated torque under extended tool wear indicates imminent overstrain failure",
      "Halt milling operations immediately and disengage spindle drive"
    ],
    "unsupported_claims": [],
    "summary": "All diagnostic claims are strictly supported by the retrieved technical manual."
  },
  "sources": [
    {
      "id": "S1",
      "source": "predictive maintanance manual.pdf",
      "page": 4,
      "text": "When torque exceeds 45 Nm with tool wear above 200 minutes, overstrain failure is imminent...",
      "relevance": 0.884
    }
  ]
}
```

---

### 4. Upload & Replace Manual
`POST /api/manual`

Uploads and activates a new maintenance manual (`.pdf` or `.docx` up to 20MB, base64-encoded).

**Request Payload:**
```json
{
  "filename": "milling_machine_service_manual_v2.pdf",
  "content": "<base64_encoded_file_string>"
}
```

---

## Troubleshooting & FAQ

### 1. `ModuleNotFoundError: No module named 'docx'`
Ensure that `python-docx` is installed inside the **exact same virtual environment** that is executing Uvicorn:
```bash
# Activate your environment first, then run:
python -m pip install -r requirements.txt
```

### 2. Missing Model Files (`FileNotFoundError`)
If `artifacts/training/models` does not contain the `best_<target>.joblib` files, execute:
```bash
python train.py
```
This will fit the candidate models, evaluate validation thresholds, and write the serialized bundles into `artifacts/training/models/`.

### 3. Groq API Key Not Configured / 503 Errors
If `/api/diagnose` returns HTTP 503:
- Verify that `.env` exists in the project root directory.
- Verify `GROQ_API_KEY` is set to a valid, active API key.
- Restart Uvicorn to ensure the updated environment variables are loaded.

### 4. Manual Upload Rejected (HTTP 400 / 413)
- Ensure the uploaded file is a standard `.pdf` or `.docx` file.
- Verify that the file size is under **20 MB**.
- Check that the PDF contains **selectable text** (scanned, image-only PDFs without an OCR layer cannot be indexed).

---

## Limitations & Industrial Disclaimers

1. **Synthetic Data Characteristics:** AI4I 2020 is a synthetic representation of milling machine operating parameters. Real industrial equipment may exhibit failure modes with non-linear, transient signatures not captured in this dataset.
2. **Extreme Class Imbalance:** Certain failure modes, specifically Tool Wear Failure (`TWF`) and Random Failure (`RNF`), possess extremely low incidence rates (fewer than 15 positive cases in 10,000 observations). While ROC-AUC scores are elevated, precision/recall metrics reflect this scarcity.
3. **Threshold Calibration Context:** The automated threshold calibration maximizes validation $F_1$-score. In production plant environments, thresholds should be aligned with business costs (e.g., cost of unexpected downtime vs. cost of unnecessary preventative tool replacement).
4. **SHAP Interpretation:** SHAP feature attributions reflect statistical associations within the trained model, which do not inherently prove physical mechanical causation.
5. **No Direct Hardware Control:** Aegis provides analytical decision support and does not include industrial PLC connectors, emergency stop (E-STOP) hardware interlocks, or safety-critical control loops.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <b>Aegis Predictive Maintenance</b> • Industrial Intelligence & Real-Time Machine Health
</p>
