# Aegis predictive maintenance

Aegis is a local FastAPI web application for exploring predictive maintenance with the AI4I 2020 dataset. It combines six trained binary classifiers, SHAP explanations, a local Chroma document index, and an optional Groq-powered diagnosis and evidence-verification workflow.

> **Research prototype only.** AI4I 2020 is synthetic data, not a representative record of your equipment. Predictions are not a safety certification or a substitute for inspection. Have qualified maintenance personnel review any recommendations before acting on them.

## What is implemented

- **Model training:** compares seven classifiers for each of six failure targets, selects a model and probability threshold using validation data, then measures final metrics on a held-out test set.
- **Interactive dashboard:** accepts machine class and five sensor readings and displays failure probabilities, classifications, risk level, and the most influential SHAP features.
- **Manual-based retrieval:** extracts PDF/DOCX text, chunks it, embeds it locally, and searches a persistent Chroma vector database.
- **Optional AI report:** sends readings, predictions, SHAP information, and retrieved manual passages to Groq to draft a diagnosis and recommended actions; a second LLM request checks claims against retrieved passages.
- **Manual replacement:** the dashboard accepts a new PDF or DOCX and activates it as the sole current manual after it has been validated and indexed.

No live sensor connection, message broker, remote vector database, OCR pipeline, or production alerting/safety integration is included.

## How the training procedure works

The training program is `train.py`; the supplied dataset is `ai4i2020.csv`.

### Data and targets

The checked-in run metadata records **10,000 rows**, no removed duplicates, and no missing sensor readings. The data is divided using a fixed random seed of `42`:

| Split | Share | Rows | Purpose |
|---|---:|---:|---|
| Training | 70% | 7,000 | Fit preprocessing and candidate models |
| Validation | 15% | 1,500 | Compare models and select each target's threshold |
| Test | 15% | 1,500 | Final evaluation after model and threshold selection |

The split is stratified on the overall `Machine failure` label. The six targets are trained as separate binary classifiers, so failure modes are not mutually exclusive:

- `Machine failure` — overall machine failure
- `TWF` — tool wear failure
- `HDF` — heat dissipation failure
- `PWF` — power failure
- `OSF` — overstrain failure
- `RNF` — random failure

### Inputs and feature preparation

The model uses machine `Type` (`L`, `M`, or `H`) and five sensor readings:

| Input | Unit |
|---|---|
| Air temperature | K |
| Process temperature | K |
| Rotational speed | rpm |
| Torque | Nm |
| Tool wear | min |

Three additional features are computed from these values:

1. Temperature difference = process temperature − air temperature.
2. Mechanical power = torque × angular speed, with angular speed calculated from rpm.
3. Torque-wear interaction = torque × tool wear.

Numeric values are median-imputed. Machine type is one-hot encoded. A scaled preprocessing path is used for the MLP; the tree-based candidates use the unscaled numeric path. Identifiers such as `UDI` and `Product ID` are not model inputs.

### Candidate models

For **each target**, the program compares:

1. Random Forest
2. Extra Trees
3. XGBoost
4. LightGBM
5. CatBoost
6. HistGradientBoosting
7. Multi-layer perceptron (MLP)

The pipeline accounts for class imbalance using class weights, sample weights, or a positive-class weight for applicable candidates. The trained dataset is particularly imbalanced: for example, the run metadata contains only 13 RNF positives in the training split and 2 RNF positives in the test split.

### How probability thresholds are selected

A classifier returns a probability; a threshold turns that probability into a binary label. The threshold is chosen **independently for each target**, not shared across all failure types:

1. Train a candidate on the training split.
2. Get positive-class probabilities for the validation split.
3. Test the 19 thresholds `0.05, 0.10, 0.15, ... 0.95`.
4. At each threshold, classify a row as positive when `probability >= threshold`, then calculate validation F1.
5. Select the candidate/threshold combination with the highest validation F1. Ties are resolved by higher validation recall, then by the lower threshold.
6. Once selected, evaluate on the test split without using test results to select a model or threshold.

F1 is the harmonic mean of precision and recall (`2 × precision × recall / (precision + recall)`). Lowering a cutoff will generally flag more cases (which can improve recall but also increase false alarms); raising it generally flags fewer cases (which can reduce false alarms but miss more failures). This procedure is a project choice for model comparison, not a clinical/operational cost optimization: it does not encode the relative cost of a missed failure versus a false alarm. The grid is coarse, and a selected threshold at `0.05` is the lowest value tested—not proof that the probability is calibrated or that `0.05` is a universally appropriate alarm setting.

The current saved artifacts record these choices:

| Target | Selected model | Validation threshold | Validation F1 | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| Machine failure | HistGradientBoosting | 0.65 | 0.8800 | 0.8600 | 0.8431 | 0.8515 |
| TWF | CatBoost | 0.45 | 0.2353 | 0.0000 | 0.0000 | 0.0000 |
| HDF | HistGradientBoosting | 0.05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| PWF | Random Forest | 0.30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| OSF | LightGBM | 0.05 | 1.0000 | 1.0000 | 0.8824 | 0.9375 |
| RNF | Extra Trees | 0.25 | 0.2500 | 0.0000 | 0.0000 | 0.0000 |

These are the values in `artifacts/training/best_models_summary.json`; retraining can change them. Perfect scores for some rare targets should not be interpreted as proof of real-world reliability. TWF and RNF have very few positive examples, and their measured results can change substantially with different splits or new data. Validate thresholds against representative, real equipment data and operational false-alarm/missed-failure costs before using predictions operationally.

### Training outputs

Training writes to `artifacts/training/`:

- `models/best_<target>.joblib` — selected model, fitted preprocessor, feature names, and selected threshold for each target.
- `best_models_summary.json` — selected model and validation/test metrics for each target.
- `model_validation_comparison.csv` — validation results for every successfully trained candidate.
- `reports/<target>_test_report.json` — test metrics and classification report.
- `reports/<target>_test_predictions.csv` — test labels, probabilities, predictions, and applied threshold.
- `run_metadata.json` — dataset counts, split information, features, targets, model list, seed, and threshold-selection rule.

## Setup on Windows

Use the Python 3.9 installation that has been used to run and validate this project. If the `py -3.9` launcher is not available, use the explicit executable path shown below.

Open PowerShell in the project folder:

```powershell
Set-Location "C:\Users\arell\OneDrive\Desktop\prdictive maintance"
```

Create and activate a virtual environment, then install dependencies:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.9` is unavailable, create the environment with the installed Python 3.9 executable:

```powershell
& "C:\Users\arell\AppData\Local\Programs\Python\Python39\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include the training libraries, FastAPI/Uvicorn, SHAP, PDF/DOCX readers, and Chroma. If Python reports `ModuleNotFoundError: No module named 'docx'`, install the project requirements into the **same Python environment used to launch Uvicorn**:

```powershell
python -m pip install -r requirements.txt
```

## Configure Groq (optional)

The local prediction endpoint does not require an LLM API key. Full diagnosis and faithfulness verification require a valid Groq key.

Create or edit `.env` in the project root:

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Replace the placeholder with your own key. Do not commit or share `.env`; it is excluded by `.gitignore`. If a real API key has been exposed, revoke it and create a replacement. Restart the server after changing `.env`.

## Train or retrain

The trained model artifacts are already supplied. Retraining is optional; it compares all seven candidate models for each target and rewrites the training outputs:

```powershell
python train.py
```

Optional arguments:

```powershell
python train.py --help
python train.py --data "C:\path\to\ai4i2020.csv" --output-dir "C:\path\to\training-output" --seed 42
```

When specifying a custom output directory, copy its `models` directory into `artifacts/training/models` (or configure the app's `MODEL_DIR`) before running the web app. The application loads the six expected model bundles from `artifacts/training/models`.

## Run the application

With the virtual environment activated:

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

If you are not using the virtual environment, launch with the known Python 3.9 executable:

```powershell
& "C:\Users\arell\AppData\Local\Programs\Python\Python39\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- Dashboard: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health/status: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

Stop the server with **Ctrl+C**. `--reload` restarts the development server when source files change; it is for local development, not production deployment.

## Use the dashboard

1. Enter a machine type (`L`, `M`, or `H`) and five readings. The UI provides example starting values and validates ranges; they are not live sensor data.
2. Select **Predict & explain** to run all six local models and show their probabilities, thresholded predictions, and SHAP feature contributions.
3. Select **Full AI diagnosis** to run the same analysis, use the model outputs and SHAP information to form a retrieval query, retrieve up to four relevant passages from the active manual, and call Groq for a diagnosis and recommended actions.
4. A second Groq request checks the draft claims against only the retrieved document evidence. Review cited passages, unsupported claims, evidence gaps, and verification status; verification is an LLM check, not a formal guarantee.
5. Use **Replace manual** to select a PDF or DOCX when you want to change the active maintenance knowledge source.

The API endpoints are:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Reports model files, active manual/index, and LLM configuration status |
| `POST` | `/api/predict` | Returns local failure probabilities, classifications, and SHAP explanations |
| `POST` | `/api/diagnose` | Returns prediction, retrieved sources, Groq diagnosis, and faithfulness check |
| `POST` | `/api/manual` | Validates and indexes a PDF/DOCX manual replacement |

The API input fields and accepted ranges are:

| API field | Allowed values |
|---|---|
| `machine_type` | `L`, `M`, or `H` |
| `air_temperature_k` | 250–350 K |
| `process_temperature_k` | 250–400 K |
| `rotational_speed_rpm` | 0–5,000 rpm |
| `torque_nm` | 0–200 Nm |
| `tool_wear_min` | 0–300 min |

Each target has its own saved threshold; a target is predicted positive when its probability is greater than or equal to that threshold. The dashboard's overall risk label is a separate, fixed rule based only on the overall `Machine failure` output: **high** when that prediction is positive and its probability is at least 0.65; **elevated** when it is positive but below 0.65; **monitor** when it is negative but at least 0.25; otherwise **low**. These risk bands are presentation heuristics, not thresholds learned during training, and should not be treated as operational alarm policy.

## Manual retrieval and vector storage

On a new/empty index, the app seeds Chroma from `predictive maintanance manual.pdf` in the project root. The similarly named DOCX supplied with the project is empty and is not a usable initial manual.

The local indexing/retrieval procedure is:

1. Extract selectable text from each PDF page or DOCX paragraphs and table rows. Scanned/image-only PDFs are not OCR-processed.
2. Split extracted text into passages of up to **180 words**, with **35 words of overlap**.
3. Generate embeddings with Chroma's local `all-MiniLM-L6-v2` embedding model.
4. Store the vectors and source/location metadata in the persistent Chroma collection `predictive_maintenance_manual`.
5. Query the active document by semantic similarity; the diagnosis workflow uses up to four retrieved passages as evidence.

No Pinecone account or hosted vector database is used. Chroma and the embedding model run locally; the first initialization may need to download the embedding model.

### Replacing a manual

- Choose a PDF or DOCX up to **20 MB** in the dashboard.
- The app checks the file type, extracts text, and indexes the new passages before switching the active-manual metadata.
- After successful activation, vectors from the previous manual are removed. A failed, empty, unreadable, encrypted, or text-free upload does not replace the active manual.
- The uploaded active file is stored in `data/manuals/`; the persistent vector database is stored in `artifacts/chroma/`. These are ignored by Git and reused after restart.
- The UI reports a cleanup warning if the new manual is active but obsolete index data could not be removed.

Do not delete `artifacts/chroma/` or `data/manuals/` if you want to preserve the currently selected uploaded manual. Keep the seed PDF in the project root as a fallback for initializing a fresh index.

## Data sharing and privacy

**Predict & explain** and manual extraction, embedding, and retrieval run locally. **Full AI diagnosis** sends the entered readings, model predictions, SHAP values, and retrieved manual passages to the configured Groq API. Use only data that you are authorized to send to that provider. Keep API keys in the local, ignored `.env` file and do not place secrets in source code or README.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'docx'`** — activate the environment used for Uvicorn, then run `python -m pip install -r requirements.txt`.
- **Models missing / run `train.py` first** — check that all six `best_*.joblib` files exist under `artifacts/training/models`; retrain with `python train.py` if they do not.
- **Groq diagnosis reports a missing key** — set `GROQ_API_KEY` in the project-root `.env` and restart Uvicorn. Local prediction can still be used without it.
- **Manual upload is rejected** — use a readable PDF or DOCX under 20 MB with extractable text. Scanned PDFs require OCR before they can be used.
- **Startup warnings from a Conda NumPy build** — use the validated Python 3.9 environment above rather than an unrelated Conda interpreter. Warnings are not necessarily an app failure, but a successful Uvicorn startup and `/api/health` response are the checks that matter.

## Important limitations

- AI4I 2020 is synthetic; it does not establish accuracy on any particular asset, sensor setup, or operating environment.
- Rare failure labels, especially TWF and RNF, have few positive samples; their metrics and selected thresholds are consequently unstable. The supplied test split has 2 RNF positive examples, so its score is not a reliable generalization estimate.
- A threshold selected by validation F1 does not reflect plant-specific risk tolerance, probability calibration, downtime cost, or safety requirements.
- SHAP describes how features influence a model output; it does not establish a physical cause of failure.
- Groq output and its evidence check may be wrong. Check the original manual and consult a qualified person.
- Real sensor acquisition, authentication, user management, monitoring, production deployment, and safety interlocks are not implemented.
