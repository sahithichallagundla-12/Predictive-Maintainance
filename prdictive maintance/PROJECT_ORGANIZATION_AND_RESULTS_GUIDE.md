# Aegis Predictive Maintenance — Project Organization & Results Guide

> **Project Root:** `c:\Users\csahi\Downloads\prdictive maintance\prdictive maintance\`  
> **Dataset:** AI4I 2020 Predictive Maintenance (10,000 instances, milling machine telemetry)  
> **Purpose:** This document is the single source of truth for the entire project directory. It maps out where trained models, multi-seed checkpoints, research evaluations, and final deliverables are stored, eliminating duplicate or disorganized clutter.

---

## 1. High-Level Folder Structure

```
prdictive maintance/
├── ai4i2020.csv                       ◄── The original dataset (10,000 rows)
├── app.py                             ◄── Interactive Web Dashboard (loads production models)
├── train.py                           ◄── Primary training pipeline (supports --seed and --output-dir)
├── preprocessed_data/                 ◄── Preprocessed datasets (scaled & unscaled)
├── static/                            ◄── Web UI assets (HTML, CSS, JS)
│
└── artifacts/                         ◄── ALL MODELS, EXPERIMENTS & RESULTS
    │
    ├── FINAL_DELIVERABLES/            ◄── ★ PRIMARY RESULTS DIRECTORY (Clean, Uncluttered, Ready)
    │   ├── Aegis_Master_Benchmark_All_Seeds.xlsx  ◄── Master Excel Workbook (6 Comprehensive Sheets)
    │   ├── Aegis_Single_Seed_Evaluation.xlsx      ◄── Single-Seed Baseline Excel Workbook
    │   ├── FINAL_RESEARCH_REPORT.md               ◄── Multi-Seed Comparative Research Study
    │   ├── FINAL_MODEL_EVALUATION_REPORT.md       ◄── Single-Seed Technical Evaluation Report
    │   ├── FINAL_RELIABILITY_AUDIT.md             ◄── 8-Point Data Leakage & Integrity Audit
    │   ├── RARE_TARGETS_TWF_RNF_REPORT.md         ◄── In-Depth Study on TWF and RNF
    │   ├── table1_main_benchmark.csv              ◄── Main Publication Table (Mean +/- Std Across 5 Seeds)
    │   ├── final_statistical_summary_all_seeds.csv◄── Flat Master Table of all 210 Seed Evaluations
    │   ├── accuracy_statistical_analysis_all_seeds.csv ◄── Accuracy Mean, Std, Var, Min, Max & 95% CIs
    │   ├── pr_curve_comparison.png                ◄── High-Resolution Precision-Recall Curves
    │   ├── confusion_matrices_all_targets.png     ◄── 6-Target Test Confusion Matrices
    │   ├── calibration_curve_machine_failure.png  ◄── Probability Calibration Curve
    │   └── roc_curves_all_targets.png             ◄── 6-Target ROC Curves
    │
    ├── training/                      ◄── PRODUCTION APP ASSETS (Seed 42)
    │   ├── models/                    ◄── 6 Production .joblib Model Bundles (Loaded by app.py)
    │   │   ├── best_machine_failure.joblib
    │   │   ├── best_hdf.joblib
    │   │   ├── best_pwf.joblib
    │   │   ├── best_osf.joblib
    │   │   ├── best_twf.joblib
    │   │   └── best_rnf.joblib
    │   ├── reports/                   ◄── Per-target test predictions and JSON reports
    │   ├── plots/                     ◄── Training validation and test plots
    │   └── final_evaluation/          ◄── Original single-seed evaluation files
    │
    ├── seeds/                         ◄── MULTI-SEED EXPERIMENTS (5 Independent Seeds)
    │   ├── 42/                        ◄── Seed 42: Models, validation summaries, predictions
    │   ├── 7/                         ◄── Seed 7:  Models, validation summaries, predictions
    │   ├── 21/                        ◄── Seed 21: Models, validation summaries, predictions
    │   ├── 99/                        ◄── Seed 99: Models, validation summaries, predictions
    │   └── 123/                       ◄── Seed 123: Models, validation summaries, predictions
    │
    ├── research/                      ◄── RAW RESEARCH RUNS & MODULAR EXPERIMENTS
    │   ├── baselines/                 ◄── All literature baseline runs (Matzka, Tuama, Waghulde) & fairness runs
    │   ├── ablations/                 ◄── 4 systematic ablations & feature margin studies
    │   ├── calibration/               ◄── Cost threshold optimization, Brier scores & architecture stacking
    │   ├── multi_seed_raw/            ◄── Raw multi-seed aggregated metrics
    │   ├── rare_targets/              ◄── TWF rule baselines, early-warning tuning & RNF permutation tests
    │   └── [Master summary CSVs & XLSX]
    │
    │
    └── chroma/                        ◄── Vector Database (RAG store for machine manuals)
```

---

## 2. Where to Find Exact Assets

### A. If you want the Final Results & Evaluation Deliverables
Go to: `artifacts/FINAL_DELIVERABLES/`
1. **Master Excel File:** [Aegis_Master_Benchmark_All_Seeds.xlsx](file:///c:/Users/csahi/Downloads/prdictive%20maintance/prdictive%20maintance/artifacts/FINAL_DELIVERABLES/Aegis_Master_Benchmark_All_Seeds.xlsx)
   * *Sheet 1: All_Seeds_Master* (All 210 seed test runs: Accuracy, Precision, Recall, F1, F2, ROC, PR, TP/FP/FN/TN)
   * *Sheet 2: Aggregate_Benchmark* (Mean $\pm$ Std and 95% Bootstrap CIs comparing Aegis to all baselines)
   * *Sheet 3: Accuracy_Statistical_Analysis* (Dedicated accuracy analysis: Mean, Std Dev, Variance, Min, Max, 95% CIs)
   * *Sheet 4: Ablation_Matrix* (Full ablation matrix across all 5 seeds)
   * *Sheet 5: Rare_Targets_TWF_RNF* (TWF early warning, wear rules, and RNF permutation proof)
   * *Sheet 6: Architecture_&_Operating_Points* (Dedicated vs Stacked vs OR-of-5-modes, cost threshold $\tau^*=0.178$)
2. **Comprehensive Research Report:** [FINAL_RESEARCH_REPORT.md](file:///c:/Users/csahi/Downloads/prdictive%20maintance/prdictive%20maintance/artifacts/FINAL_DELIVERABLES/FINAL_RESEARCH_REPORT.md)
3. **Single-Seed Project Evaluation Report:** [FINAL_MODEL_EVALUATION_REPORT.md](file:///c:/Users/csahi/Downloads/prdictive%20maintance/prdictive%20maintance/artifacts/FINAL_DELIVERABLES/FINAL_MODEL_EVALUATION_REPORT.md)
4. **Reliability & Leakage Audit:** [FINAL_RELIABILITY_AUDIT.md](file:///c:/Users/csahi/Downloads/prdictive%20maintance/prdictive%20maintance/artifacts/FINAL_DELIVERABLES/FINAL_RELIABILITY_AUDIT.md)

---

### B. If you want the Trained Models
1. **Production Models (Loaded by Dashboard `app.py`):**
   * Stored in: `artifacts/training/models/`
   * Files:
     * `best_machine_failure.joblib` (HistGradientBoosting, $\tau = 0.65$)
     * `best_hdf.joblib` (HistGradientBoosting, $\tau = 0.05$)
     * `best_pwf.joblib` (RandomForest, $\tau = 0.30$)
     * `best_osf.joblib` (LightGBM, $\tau = 0.05$)
     * `best_twf.joblib` (CatBoost, $\tau = 0.45$)
     * `best_rnf.joblib` (ExtraTrees, $\tau = 0.25$)
   * *Note:* Each `.joblib` bundle is self-contained, storing the fitted scikit-learn `estimator`, the training-fit `preprocessor`, the feature names, and the validation-tuned `threshold`.
2. **Multi-Seed Checkpoints:**
   * Stored in: `artifacts/seeds/{seed}/models/` for seeds **42, 7, 21, 99, 123**.

---

### C. If you want the Preprocessed Data
Stored in: `preprocessed_data/`
* `preprocessed_scaled.csv` — Features with StandardScaling (used for SVM, DNN, Logistic Regression).
* `preprocessed_tree.csv` — Features with unscaled numerical values (used for Tree ensembles, LightGBM, CatBoost, XGBoost).

---

### D. If you want to Run or Replicate Experiments
All scripts in the project root are modular and purpose-built:
* `python app.py` — Launches the live interactive predictive maintenance web dashboard.
* `python train.py --seed <seed> --output-dir <dir>` — Trains the 6 targets with per-target model selection.
* `python run_all_baselines.py --seed <seed>` — Runs published literature baselines (Matzka, Tuama, Waghulde) on identical splits.
* `python evaluate_project.py` — Orchestrates multi-seed aggregation, ablations, and architectures.
* `python evaluate_rare_targets.py` — Computes TWF early-warning models, wear rules, and RNF permutation tests.
* `python build_final_statistical_summary.py` — Re-compiles the master Excel workbook and CSV tables.

---

## 3. Summary of Project Clean-Up Actions Taken
1. **Removed Root Artifact Shadow:** Deleted an empty temporary `artifacts/seeds/42` folder from the parent directory to prevent directory confusion.
2. **Consolidated Final Deliverables:** Created `artifacts/FINAL_DELIVERABLES/` which houses all primary spreadsheets, markdown reports, publication tables, and high-resolution figures in one clean location.
3. **Preserved Core Pipelines:** Maintained all original paths in `artifacts/training/` and `artifacts/seeds/` so that existing scripts (`app.py`, `train.py`) continue to run seamlessly without broken dependencies.
4. **Cleaned Temporary Scripts:** Removed intermediate one-off scripts (such as `append_accuracy_sheet.py`), leaving only clean, maintainable runner scripts in the root.
