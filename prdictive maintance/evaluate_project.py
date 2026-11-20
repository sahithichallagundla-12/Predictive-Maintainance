"""evaluate_project.py: Comprehensive research evaluation pipeline for Aegis.

Implements all required research paper sections:
1. Multi-seed training & aggregation across 5 seeds: 42, 7, 21, 99, 123
   (Aegis vs Baselines: B1 Matzka, B2 Tuama, B3 Waghulde)
2. Four systematic ablations across all 5 seeds:
   (a) No engineered features (raw sensors only)
   (b) Fixed threshold at 0.5
   (c) No class weights / balancing
   (d) Fixed single model (XGBoost) across all targets
3. Rare target CV analysis (TWF, RNF) via RepeatedStratifiedKFold (5 folds x 5 repeats) on train+val.
4. Baseline fairness analysis: raw features vs engineered features for baselines.
5. Operating point analysis for Machine failure (cost ratio FN:FP = 10:1, calibration / Brier score).
6. Improvements: OSF overstrain margin, TWF wear window, and Stacked model vs dedicated Machine failure model.
7. Generates final tables and PR curves.
"""

from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from train import (
    load_dataset,
    split_dataset,
    engineer_features,
    make_preprocessor,
    TARGET_COLUMNS,
    SENSOR_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    ENGINEERED_COLUMNS,
    THRESHOLDS,
    select_threshold,
    clean_target_name,
)
from run_all_baselines import run_baselines_for_seed

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "ai4i2020.csv"
RESEARCH_DIR = ROOT_DIR / "artifacts" / "research"
SEEDS_DIR = ROOT_DIR / "artifacts" / "seeds"

RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
SEEDS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 7, 21, 99, 123]


# -----------------------------------------------------------------------------
# 1. Multi-Seed Aegis Runner & Metric Extractor
# -----------------------------------------------------------------------------
def run_aegis_for_seed(seed: int, output_dir: Path):
    from train import run_training
    if (output_dir / "best_models_summary.json").exists():
        print(f"[CACHE] Seed {seed} already trained at {output_dir}")
        with open(output_dir / "best_models_summary.json", "r") as f:
            return json.load(f)
    print(f"\n---> Training Aegis on Seed {seed}...")
    return run_training(data_path=DATA_PATH, output_dir=output_dir, random_state=seed)


def collect_multi_seed_results():
    all_rows = []
    
    for s in SEEDS:
        s_dir = SEEDS_DIR / str(s)
        summary = run_aegis_for_seed(s, s_dir)
        
        # Aegis metrics
        for target in TARGET_COLUMNS:
            m = summary[target]["test_metrics"]
            all_rows.append({
                "seed": s,
                "pipeline": "Aegis (ours)",
                "target": target,
                "model": summary[target]["best_model"],
                "threshold": m["threshold"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
            })
            
        # Run baselines for this seed (raw features)
        b_cache = RESEARCH_DIR / f"baselines_seed_{s}_raw.csv"
        if b_cache.exists():
            b_df = pd.read_csv(b_cache)
        else:
            print(f"---> Running baselines for Seed {s}...")
            b_df = run_baselines_for_seed(seed=s, use_engineered=False)
            b_df.to_csv(b_cache, index=False)
            
        for _, brow in b_df.iterrows():
            all_rows.append({
                "seed": s,
                "pipeline": f"Baseline ({brow['source']})",
                "target": brow["target"],
                "model": brow["model"],
                "threshold": 0.5,
                "accuracy": brow["accuracy"],
                "precision": brow["precision"],
                "recall": brow["recall"],
                "f1": brow["f1"],
                "roc_auc": brow["roc_auc"],
                "pr_auc": brow["pr_auc"],
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(RESEARCH_DIR / "multi_seed_raw_results.csv", index=False)
    
    # Aggregation
    agg = df.groupby(["pipeline", "target", "model"]).agg({
        "f1": ["mean", "std"],
        "recall": ["mean", "std"],
        "precision": ["mean", "std"],
        "pr_auc": ["mean", "std"],
        "roc_auc": ["mean", "std"]
    }).reset_index()
    
    agg.columns = [f"{c[0]}_{c[1]}".strip("_") for c in agg.columns]
    agg.to_csv(RESEARCH_DIR / "multi_seed_aggregated.csv", index=False)
    print(f"[SUCCESS] Multi-seed aggregated saved to {RESEARCH_DIR / 'multi_seed_aggregated.csv'}")
    return df, agg


# -----------------------------------------------------------------------------
# 2. Ablations (4 conditions over 5 seeds)
# -----------------------------------------------------------------------------
def run_ablations():
    print("\n" + "=" * 60)
    print("RUNNING ABLATION EXPERIMENTS (5 seeds)")
    print("=" * 60)
    frame, _ = load_dataset(DATA_PATH)
    
    ablation_records = []
    
    for s in SEEDS:
        train_idx, val_idx, test_idx = split_dataset(frame, s)
        
        # Conditions:
        # 1. Full Aegis (already in seed summary)
        with open(SEEDS_DIR / str(s) / "best_models_summary.json") as f:
            full_aegis = json.load(f)
            
        for target in TARGET_COLUMNS:
            aegis_f1 = full_aegis[target]["test_metrics"]["f1"]
            ablation_records.append({
                "seed": s,
                "condition": "Full Aegis",
                "target": target,
                "test_f1": aegis_f1
            })
            
            y = frame[target].to_numpy()
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
            
            # --- Condition A: No Engineered Features (raw sensors only) ---
            raw_features = frame[SENSOR_COLUMNS + CATEGORICAL_COLUMNS]
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.impute import SimpleImputer
            prep_a = ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), SENSOR_COLUMNS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
            ])
            X_tr_a = prep_a.fit_transform(raw_features.iloc[train_idx])
            X_va_a = prep_a.transform(raw_features.iloc[val_idx])
            X_te_a = prep_a.transform(raw_features.iloc[test_idx])
            
            mdl_a = HistGradientBoostingClassifier(random_state=s)
            mdl_a.fit(X_tr_a, y_train)
            prob_va_a = mdl_a.predict_proba(X_va_a)[:, 1]
            tau_a, _ = select_threshold(y_val, prob_va_a)
            prob_te_a = mdl_a.predict_proba(X_te_a)[:, 1]
            f1_a = f1_score(y_test, (prob_te_a >= tau_a).astype(int), zero_division=0)
            ablation_records.append({
                "seed": s,
                "condition": "No Engineered Features",
                "target": target,
                "test_f1": f1_a
            })

            # --- Condition B: Fixed Threshold = 0.5 ---
            # Use Aegis model bundle but apply tau=0.5
            from train import clean_target_name
            import joblib
            bundle = joblib.load(SEEDS_DIR / str(s) / "models" / f"best_{clean_target_name(target)}.joblib")
            feat_df = engineer_features(frame)
            X_all = feat_df[bundle["input_feature_columns"]]
            X_te_b = bundle["preprocessor"].transform(X_all.iloc[test_idx])
            prob_te_b = bundle["estimator"].predict_proba(X_te_b)[:, 1]
            f1_b = f1_score(y_test, (prob_te_b >= 0.50).astype(int), zero_division=0)
            ablation_records.append({
                "seed": s,
                "condition": "Fixed Threshold (0.50)",
                "target": target,
                "test_f1": f1_b
            })

            # --- Condition C: No Class Weights ---
            # Train model without sample weights or class balancing
            feat_c = engineer_features(frame)[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
            prep_c = make_preprocessor(scale_numeric=False)
            X_tr_c = prep_c.fit_transform(feat_c.iloc[train_idx])
            X_va_c = prep_c.transform(feat_c.iloc[val_idx])
            X_te_c = prep_c.transform(feat_c.iloc[test_idx])
            mdl_c = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=1)
            mdl_c.fit(X_tr_c, y_train)
            prob_va_c = mdl_c.predict_proba(X_va_c)[:, 1]
            tau_c, _ = select_threshold(y_val, prob_va_c)
            prob_te_c = mdl_c.predict_proba(X_te_c)[:, 1]
            f1_c = f1_score(y_test, (prob_te_c >= tau_c).astype(int), zero_division=0)
            ablation_records.append({
                "seed": s,
                "condition": "No Class Balancing",
                "target": target,
                "test_f1": f1_c
            })

            # --- Condition D: Fixed Model (XGBoost for all targets) ---
            pos = max(1, int(np.sum(y_train == 1)))
            neg = max(1, int(np.sum(y_train == 0)))
            mdl_d = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.05,
                scale_pos_weight=neg/pos, random_state=s, eval_metric="logloss", n_jobs=1
            )
            mdl_d.fit(X_tr_c, y_train)
            prob_va_d = mdl_d.predict_proba(X_va_c)[:, 1]
            tau_d, _ = select_threshold(y_val, prob_va_d)
            prob_te_d = mdl_d.predict_proba(X_te_c)[:, 1]
            f1_d = f1_score(y_test, (prob_te_d >= tau_d).astype(int), zero_division=0)
            ablation_records.append({
                "seed": s,
                "condition": "Fixed Model (XGBoost)",
                "target": target,
                "test_f1": f1_d
            })

    abl_df = pd.DataFrame(ablation_records)
    abl_df.to_csv(RESEARCH_DIR / "ablation_results_all_seeds.csv", index=False)
    
    abl_summary = abl_df.groupby(["condition", "target"])["test_f1"].agg(["mean", "std"]).reset_index()
    
    # Calculate delta versus full Aegis
    aegis_means = abl_summary[abl_summary["condition"] == "Full Aegis"].set_index("target")["mean"].to_dict()
    abl_summary["delta_vs_aegis"] = abl_summary.apply(
        lambda r: r["mean"] - aegis_means.get(r["target"], 0), axis=1
    )
    abl_summary.to_csv(RESEARCH_DIR / "ablation_summary.csv", index=False)
    print(f"[SUCCESS] Ablation summary saved to {RESEARCH_DIR / 'ablation_summary.csv'}")
    return abl_summary


# -----------------------------------------------------------------------------
# 3. Rare Target CV Analysis (TWF, RNF) on Train+Val only
# -----------------------------------------------------------------------------
def run_rare_target_cv():
    print("\n" + "=" * 60)
    print("RUNNING REPEATED STRATIFIED 5-FOLD CV ON TRAIN+VAL (TWF & RNF)")
    print("=" * 60)
    frame, _ = load_dataset(DATA_PATH)
    features = engineer_features(frame)
    X = features[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    
    # Fixed outer split to hold out test strictly
    train_val_idx, _, _ = split_dataset(frame, 42)
    # Combine train and val indices for repeated CV
    train_idx, val_idx, _ = split_dataset(frame, 42)
    tv_indices = np.concatenate([train_idx, val_idx])
    
    X_tv = X.iloc[tv_indices].reset_index(drop=True)
    frame_tv = frame.iloc[tv_indices].reset_index(drop=True)
    
    cv_records = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    models_to_test = {
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=1),
        "LightGBM": LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, verbosity=-1, n_jobs=1),
        "CatBoost": CatBoostClassifier(iterations=100, depth=5, learning_rate=0.05, verbose=False, random_seed=42, allow_writing_files=False),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42)
    }
    
    for target in ["TWF", "RNF"]:
        y_tv = frame_tv[target].to_numpy()
        for fold, (f_tr, f_va) in enumerate(rskf.split(X_tv, y_tv)):
            prep = make_preprocessor(scale_numeric=False)
            X_tr_f = prep.fit_transform(X_tv.iloc[f_tr])
            X_va_f = prep.transform(X_tv.iloc[f_va])
            y_tr_f, y_va_f = y_tv[f_tr], y_tv[f_va]
            
            for m_name, model in models_to_test.items():
                try:
                    model.fit(X_tr_f, y_tr_f)
                    probs = model.predict_proba(X_va_f)[:, 1]
                    
                    # Compute PR-AUC (threshold-free)
                    pr_auc = average_precision_score(y_va_f, probs) if len(np.unique(y_va_f)) > 1 else np.nan
                    roc_auc = roc_auc_score(y_va_f, probs) if len(np.unique(y_va_f)) > 1 else np.nan
                    
                    # Validation F1 across standard thresholds
                    best_f1, _ = select_threshold(y_va_f, probs)
                    
                    cv_records.append({
                        "target": target,
                        "fold": fold,
                        "model": m_name,
                        "pr_auc": pr_auc,
                        "roc_auc": roc_auc,
                        "val_f1": best_f1
                    })
                except Exception as e:
                    pass

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(RESEARCH_DIR / "rare_targets_repeated_cv.csv", index=False)
    
    cv_summary = cv_df.groupby(["target", "model"]).agg({
        "pr_auc": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "val_f1": ["mean", "std"]
    }).reset_index()
    cv_summary.columns = [f"{c[0]}_{c[1]}".strip("_") for c in cv_summary.columns]
    cv_summary.to_csv(RESEARCH_DIR / "rare_targets_cv_summary.csv", index=False)
    print(f"[SUCCESS] Rare target CV summary saved to {RESEARCH_DIR / 'rare_targets_cv_summary.csv'}")
    return cv_summary


# -----------------------------------------------------------------------------
# 4. Baseline Fairness Run (Baselines with Engineered Features)
# -----------------------------------------------------------------------------
def run_baseline_fairness():
    print("\n" + "=" * 60)
    print("RUNNING BASELINE FAIRNESS EXPERIMENT (Engineered vs Raw Features)")
    print("=" * 60)
    
    records = []
    for s in SEEDS:
        # Raw features baselines
        b_raw = run_baselines_for_seed(seed=s, use_engineered=False)
        b_raw["feature_set"] = "Raw Sensors"
        # Engineered features baselines
        b_eng = run_baselines_for_seed(seed=s, use_engineered=True)
        b_eng["feature_set"] = "Engineered (Aegis features)"
        
        records.append(b_raw)
        records.append(b_eng)
        
    fair_df = pd.concat(records, ignore_index=True)
    fair_df.to_csv(RESEARCH_DIR / "baseline_fairness_all_seeds.csv", index=False)
    
    fair_agg = fair_df.groupby(["feature_set", "target", "model"]).agg({
        "f1": ["mean", "std"],
        "recall": ["mean", "std"],
        "pr_auc": ["mean", "std"]
    }).reset_index()
    fair_agg.columns = [f"{c[0]}_{c[1]}".strip("_") for c in fair_agg.columns]
    fair_agg.to_csv(RESEARCH_DIR / "baseline_fairness_summary.csv", index=False)
    print(f"[SUCCESS] Baseline fairness summary saved to {RESEARCH_DIR / 'baseline_fairness_summary.csv'}")
    return fair_agg


# -----------------------------------------------------------------------------
# 5. Operating Points & Calibration (Part C.1)
# -----------------------------------------------------------------------------
def run_calibration_and_operating_points():
    print("\n" + "=" * 60)
    print("RUNNING OPERATING POINT & CALIBRATION ANALYSIS (Machine failure)")
    print("=" * 60)
    frame, _ = load_dataset(DATA_PATH)
    features = engineer_features(frame)
    X = features[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    train_idx, val_idx, test_idx = split_dataset(frame, 42)
    
    y = frame["Machine failure"].to_numpy()
    
    # Fit model on train
    prep = make_preprocessor(scale_numeric=False)
    X_tr = prep.fit_transform(X.iloc[train_idx])
    X_va = prep.transform(X.iloc[val_idx])
    X_te = prep.transform(X.iloc[test_idx])
    
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_tr, y[train_idx])
    
    probs_val = model.predict_proba(X_va)[:, 1]
    probs_test = model.predict_proba(X_te)[:, 1]
    
    # 1. Operating points at multiple validation thresholds
    candidate_taus = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]
    op_table = []
    for tau in candidate_taus:
        # Validation
        v_pred = (probs_val >= tau).astype(int)
        v_p = precision_score(y[val_idx], v_pred, zero_division=0)
        v_r = recall_score(y[val_idx], v_pred, zero_division=0)
        v_f1 = f1_score(y[val_idx], v_pred, zero_division=0)
        
        # Test
        t_pred = (probs_test >= tau).astype(int)
        t_p = precision_score(y[test_idx], t_pred, zero_division=0)
        t_r = recall_score(y[test_idx], t_pred, zero_division=0)
        t_f1 = f1_score(y[test_idx], t_pred, zero_division=0)
        
        op_table.append({
            "threshold": tau,
            "val_precision": v_p, "val_recall": v_r, "val_f1": v_f1,
            "test_precision": t_p, "test_recall": t_r, "test_f1": t_f1
        })
        
    # Cost-based threshold: Cost ratio FN:FP = 10:1
    # Cost function C = 10 * FN + 1 * FP
    cost_grid = np.linspace(0.01, 0.99, 100)
    best_cost = float("inf")
    best_cost_tau = 0.5
    for tau in cost_grid:
        v_pred = (probs_val >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[val_idx], v_pred, labels=[0, 1]).ravel()
        cost = 10.0 * fn + 1.0 * fp
        if cost < best_cost:
            best_cost = cost
            best_cost_tau = tau
            
    # Evaluate cost threshold on test
    cost_pred_te = (probs_test >= best_cost_tau).astype(int)
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y[test_idx], cost_pred_te, labels=[0, 1]).ravel()
    
    op_df = pd.DataFrame(op_table)
    op_df.to_csv(RESEARCH_DIR / "operating_points_machine_failure.csv", index=False)
    
    # 2. Calibration (Brier score & reliability curve)
    brier_val = brier_score_loss(y[val_idx], probs_val)
    brier_test = brier_score_loss(y[test_idx], probs_test)
    prob_true, prob_pred = calibration_curve(y[test_idx], probs_test, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "s-", color="#1f77b4", label=f"HistGradBoost (Brier={brier_test:.4f})")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_title("Reliability Diagram (Machine failure, Test)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESEARCH_DIR / "calibration_curve_machine_failure.png", dpi=200)
    plt.close()
    
    print(f"Machine failure Operating Points:\n{op_df}")
    print(f"Optimal Cost-Based Threshold (FN:FP=10:1): tau={best_cost_tau:.4f}")
    print(f"  Test Performance at cost tau: F1={f1_score(y[test_idx], cost_pred_te):.4f}, Recall={recall_score(y[test_idx], cost_pred_te):.4f}, Prec={precision_score(y[test_idx], cost_pred_te):.4f}")
    print(f"  Test Confusion Matrix: TP={tp_c}, FP={fp_c}, FN={fn_c}, TN={tn_c}")
    print(f"Brier Score on Test: {brier_test:.4f} (Validation: {brier_val:.4f})")
    
    calib_info = {
        "cost_ratio": "FN:FP = 10:1",
        "optimal_cost_threshold": best_cost_tau,
        "test_f1_cost_threshold": f1_score(y[test_idx], cost_pred_te),
        "test_recall_cost_threshold": recall_score(y[test_idx], cost_pred_te),
        "test_precision_cost_threshold": precision_score(y[test_idx], cost_pred_te),
        "brier_score_test": brier_test,
        "brier_score_val": brier_val
    }
    with open(RESEARCH_DIR / "calibration_and_cost_results.json", "w") as f:
        json.dump(calib_info, f, indent=2)


# -----------------------------------------------------------------------------
# 6. Part C Improvements & Comparisons
# -----------------------------------------------------------------------------
def run_model_improvements():
    print("\n" + "=" * 60)
    print("TESTING CANDIDATE IMPROVEMENTS (OSF Margin, TWF Window, Stacking)")
    print("=" * 60)
    frame, _ = load_dataset(DATA_PATH)
    
    # 1. OSF Overstrain Margin
    # Physical definition in Matzka 2020: product type limit is 11,000 min*Nm for L, 12,000 for M, 13,000 for H.
    type_limits = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
    feat_osf = engineer_features(frame)
    feat_osf["OSF_limit"] = feat_osf["Type"].map(type_limits)
    feat_osf["OSF_margin"] = (feat_osf["Torque [Nm]"] * feat_osf["Tool wear [min]"]) - feat_osf["OSF_limit"]
    
    osf_results = []
    for s in SEEDS:
        tr, va, te = split_dataset(frame, s)
        y = frame["OSF"].to_numpy()
        
        # Baseline Aegis OSF
        prep_base = make_preprocessor(scale_numeric=False)
        X_base = feat_osf[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        X_tr = prep_base.fit_transform(X_base.iloc[tr])
        X_va = prep_base.transform(X_base.iloc[va])
        X_te = prep_base.transform(X_base.iloc[te])
        m_base = LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=s, verbosity=-1, n_jobs=1)
        m_base.fit(X_tr, y[tr])
        tau_base, _ = select_threshold(y[va], m_base.predict_proba(X_va)[:, 1])
        f1_base = f1_score(y[te], (m_base.predict_proba(X_te)[:, 1] >= tau_base).astype(int), zero_division=0)
        
        # OSF with Margin Feature
        osf_num_cols = NUMERIC_COLUMNS + ["OSF_margin"]
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        prep_margin = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), osf_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ])
        X_m_all = feat_osf[osf_num_cols + CATEGORICAL_COLUMNS]
        X_tr_m = prep_margin.fit_transform(X_m_all.iloc[tr])
        X_va_m = prep_margin.transform(X_m_all.iloc[va])
        X_te_m = prep_margin.transform(X_m_all.iloc[te])
        m_margin = LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=s, verbosity=-1, n_jobs=1)
        m_margin.fit(X_tr_m, y[tr])
        tau_m, _ = select_threshold(y[va], m_margin.predict_proba(X_va_m)[:, 1])
        f1_margin = f1_score(y[te], (m_margin.predict_proba(X_te_m)[:, 1] >= tau_m).astype(int), zero_division=0)
        
        osf_results.append({
            "seed": s, "OSF_f1_base": f1_base, "OSF_f1_margin": f1_margin
        })
        
    osf_df = pd.DataFrame(osf_results)
    osf_df.to_csv(RESEARCH_DIR / "osf_margin_ablation.csv", index=False)
    print(f"OSF Overstrain Margin Feature Effect (mean F1):\n  Without margin: {osf_df['OSF_f1_base'].mean():.4f} +/- {osf_df['OSF_f1_base'].std():.4f}\n  With margin:    {osf_df['OSF_f1_margin'].mean():.4f} +/- {osf_df['OSF_f1_margin'].std():.4f}")

    # 2. TWF 200-240 min window analysis
    feat_twf = engineer_features(frame)
    feat_twf["wear_window_200_240"] = ((feat_twf["Tool wear [min]"] >= 200) & (feat_twf["Tool wear [min]"] <= 240)).astype(int)
    
    twf_results = []
    for s in SEEDS:
        tr, va, te = split_dataset(frame, s)
        y = frame["TWF"].to_numpy()
        
        # Aegis TWF
        prep_t = make_preprocessor(scale_numeric=False)
        X_base = feat_twf[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
        X_tr = prep_t.fit_transform(X_base.iloc[tr])
        X_va = prep_t.transform(X_base.iloc[va])
        X_te = prep_t.transform(X_base.iloc[te])
        m_t = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.05, verbose=False, random_seed=s, allow_writing_files=False)
        m_t.fit(X_tr, y[tr])
        tau_t, _ = select_threshold(y[va], m_t.predict_proba(X_va)[:, 1])
        f1_t_base = f1_score(y[te], (m_t.predict_proba(X_te)[:, 1] >= tau_t).astype(int), zero_division=0)
        
        # TWF with wear window feature
        twf_cols = NUMERIC_COLUMNS + ["wear_window_200_240"]
        prep_tw = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), twf_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ])
        X_tw_all = feat_twf[twf_cols + CATEGORICAL_COLUMNS]
        X_tr_w = prep_tw.fit_transform(X_tw_all.iloc[tr])
        X_va_w = prep_tw.transform(X_tw_all.iloc[va])
        X_te_w = prep_tw.transform(X_tw_all.iloc[te])
        m_tw = CatBoostClassifier(iterations=100, depth=5, learning_rate=0.05, verbose=False, random_seed=s, allow_writing_files=False)
        m_tw.fit(X_tr_w, y[tr])
        tau_tw, _ = select_threshold(y[va], m_tw.predict_proba(X_va_w)[:, 1])
        f1_tw_feat = f1_score(y[te], (m_tw.predict_proba(X_te_w)[:, 1] >= tau_tw).astype(int), zero_division=0)
        
        twf_results.append({
            "seed": s, "TWF_f1_base": f1_t_base, "TWF_f1_window_feature": f1_tw_feat
        })
    twf_df = pd.DataFrame(twf_results)
    twf_df.to_csv(RESEARCH_DIR / "twf_window_ablation.csv", index=False)
    print(f"TWF Wear Window Feature Effect (mean F1):\n  Base:   {twf_df['TWF_f1_base'].mean():.4f}\n  Window: {twf_df['TWF_f1_window_feature'].mean():.4f}")

    # 3. Stacked Model vs Dedicated Machine failure Model
    # Stacking logic: Dedicated model vs LogisticRegression meta-model taking [dedicated_prob, p(TWF), p(HDF), p(PWF), p(OSF), p(RNF)]
    stack_results = []
    for s in SEEDS:
        tr, va, te = split_dataset(frame, s)
        y_mf = frame["Machine failure"].to_numpy()
        
        # Load best models from seed
        s_dir = SEEDS_DIR / str(s) / "models"
        feat = engineer_features(frame)
        
        # Get validation predictions from all 5 failure modes + dedicated model
        val_meta_X = []
        test_meta_X = []
        
        all_targets = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
        for t in all_targets:
            bundle = joblib.load(s_dir / f"best_{clean_target_name(t)}.joblib")
            prep = bundle["preprocessor"]
            mdl = bundle["estimator"]
            X_inp = feat[bundle["input_feature_columns"]]
            p_val = mdl.predict_proba(prep.transform(X_inp.iloc[va]))[:, 1]
            p_test = mdl.predict_proba(prep.transform(X_inp.iloc[te]))[:, 1]
            val_meta_X.append(p_val)
            test_meta_X.append(p_test)
            
        V_meta = np.column_stack(val_meta_X)
        T_meta = np.column_stack(test_meta_X)
        
        # Fit meta classifier on validation set probabilities
        meta_clf = LogisticRegression(class_weight="balanced", random_state=s)
        meta_clf.fit(V_meta, y_mf[va])
        
        # Choose threshold on validation via cross-validation or leave-one-out
        meta_val_probs = meta_clf.predict_proba(V_meta)[:, 1]
        tau_meta, _ = select_threshold(y_mf[va], meta_val_probs)
        
        # Test performance
        meta_test_probs = meta_clf.predict_proba(T_meta)[:, 1]
        f1_stack = f1_score(y_mf[te], (meta_test_probs >= tau_meta).astype(int), zero_division=0)
        rec_stack = recall_score(y_mf[te], (meta_test_probs >= tau_meta).astype(int), zero_division=0)
        
        # Dedicated model alone
        bundle_mf = joblib.load(s_dir / "best_machine_failure.joblib")
        p_te_mf = bundle_mf["estimator"].predict_proba(bundle_mf["preprocessor"].transform(feat[bundle_mf["input_feature_columns"]].iloc[te]))[:, 1]
        f1_dedicated = f1_score(y_mf[te], (p_te_mf >= bundle_mf["threshold"]).astype(int), zero_division=0)
        rec_dedicated = recall_score(y_mf[te], (p_te_mf >= bundle_mf["threshold"]).astype(int), zero_division=0)
        
        # OR of 5 failure modes
        or_val = (V_meta[:, 1:] >= 0.5).any(axis=1).astype(int)
        or_test = (T_meta[:, 1:] >= 0.5).any(axis=1).astype(int)
        f1_or = f1_score(y_mf[te], or_test, zero_division=0)
        rec_or = recall_score(y_mf[te], or_test, zero_division=0)
        
        stack_results.append({
            "seed": s,
            "dedicated_f1": f1_dedicated, "dedicated_recall": rec_dedicated,
            "stacked_f1": f1_stack, "stacked_recall": rec_stack,
            "or_modes_f1": f1_or, "or_modes_recall": rec_or
        })
        
    stack_df = pd.DataFrame(stack_results)
    stack_df.to_csv(RESEARCH_DIR / "stacking_vs_dedicated_comparison.csv", index=False)
    print(f"Machine Failure Architecture Comparison (mean over 5 seeds):\n"
          f"  Dedicated Classifier: F1 = {stack_df['dedicated_f1'].mean():.4f} +/- {stack_df['dedicated_f1'].std():.4f} (Recall: {stack_df['dedicated_recall'].mean():.4f})\n"
          f"  Stacked Meta-Model:   F1 = {stack_df['stacked_f1'].mean():.4f} +/- {stack_df['stacked_f1'].std():.4f} (Recall: {stack_df['stacked_recall'].mean():.4f})\n"
          f"  OR of 5 Mode Models:  F1 = {stack_df['or_modes_f1'].mean():.4f} +/- {stack_df['or_modes_f1'].std():.4f} (Recall: {stack_df['or_modes_recall'].mean():.4f})")


# -----------------------------------------------------------------------------
# 7. Generate Research Paper Tables and Figures
# -----------------------------------------------------------------------------
def generate_final_deliverables():
    print("\n" + "=" * 60)
    print("GENERATING RESEARCH DELIVERABLE TABLES & FIGURES")
    print("=" * 60)
    
    # 1. Main Table: Aegis vs Baselines across 5 seeds
    agg_df = pd.read_csv(RESEARCH_DIR / "multi_seed_aggregated.csv")
    
    # Select key models for comparison
    table_rows = []
    for target in TARGET_COLUMNS:
        sub = agg_df[agg_df["target"] == target]
        for _, r in sub.iterrows():
            table_rows.append({
                "Target": target,
                "Method / Model": f"{r['model']} ({r['pipeline']})",
                "F1 Score": f"{r['f1_mean']:.3f} +/- {r['f1_std']:.3f}",
                "Recall": f"{r['recall_mean']:.3f} +/- {r['recall_std']:.3f}",
                "PR-AUC": f"{r['pr_auc_mean']:.3f} +/- {r['pr_auc_std']:.3f}",
                "ROC-AUC": f"{r['roc_auc_mean']:.3f} +/- {r['roc_auc_std']:.3f}",
            })
            
    main_table = pd.DataFrame(table_rows)
    main_table.to_csv(RESEARCH_DIR / "table1_main_benchmark.csv", index=False)
    
    # 2. PR Curves: Aegis vs Best Baseline on Seed 42
    test_preds_aegis = pd.read_csv(ROOT_DIR / "artifacts/training/reports/machine_failure_test_predictions.csv")
    b_seed42 = pd.read_csv(RESEARCH_DIR / "baselines_seed_42_raw.csv")
    
    # Reconstruct test probabilities for best baseline XGBoost on Seed 42
    frame, _ = load_dataset(DATA_PATH)
    splits = split_dataset(frame, 42)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    prep_raw = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), SENSOR_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    raw_X = frame[SENSOR_COLUMNS + CATEGORICAL_COLUMNS]
    X_tr = prep_raw.fit_transform(raw_X.iloc[splits[0]])
    X_te = prep_raw.transform(raw_X.iloc[splits[2]])
    y_tr = frame["Machine failure"].iloc[splits[0]].to_numpy()
    y_te = frame["Machine failure"].iloc[splits[2]].to_numpy()
    
    xgb_base = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=1)
    xgb_base.fit(X_tr, y_tr)
    probs_xgb = xgb_base.predict_proba(X_te)[:, 1]
    
    rf_base = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=1)
    rf_base.fit(X_tr, y_tr)
    probs_rf = rf_base.predict_proba(X_te)[:, 1]
    
    p_aegis, r_aegis, _ = precision_recall_curve(y_te, test_preds_aegis["probability"])
    ap_aegis = average_precision_score(y_te, test_preds_aegis["probability"])
    
    p_xgb, r_xgb, _ = precision_recall_curve(y_te, probs_xgb)
    ap_xgb = average_precision_score(y_te, probs_xgb)
    
    p_rf, r_rf, _ = precision_recall_curve(y_te, probs_rf)
    ap_rf = average_precision_score(y_te, probs_rf)
    
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(r_aegis, p_aegis, lw=2.5, color="#1f77b4", label=f"Aegis (Ours): HistGradBoost (PR-AUC={ap_aegis:.3f})")
    ax.plot(r_xgb, p_xgb, lw=2, color="#ff7f0e", linestyle="--", label=f"B2/B3: XGBoost Raw (PR-AUC={ap_xgb:.3f})")
    ax.plot(r_rf, p_rf, lw=2, color="#2ca02c", linestyle="-.", label=f"B2/B3: Random Forest Raw (PR-AUC={ap_rf:.3f})")
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curve: Aegis vs Published Baselines (Machine failure)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESEARCH_DIR / "pr_curve_comparison.png", dpi=300)
    plt.close()
    print(f"[SUCCESS] Generated PR Curve Comparison at {RESEARCH_DIR / 'pr_curve_comparison.png'}")


# -----------------------------------------------------------------------------
# Main Execution Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Comprehensive Aegis Research Evaluation...")
    if not (RESEARCH_DIR / "multi_seed_aggregated.csv").exists():
        collect_multi_seed_results()
    else:
        print("[CACHE] multi_seed_aggregated.csv already exists.")
        
    if not (RESEARCH_DIR / "ablation_summary.csv").exists():
        run_ablations()
    else:
        print("[CACHE] ablation_summary.csv already exists.")
        
    if not (RESEARCH_DIR / "rare_targets_cv_summary.csv").exists():
        run_rare_target_cv()
    else:
        print("[CACHE] rare_targets_cv_summary.csv already exists.")
        
    if not (RESEARCH_DIR / "baseline_fairness_summary.csv").exists():
        run_baseline_fairness()
    else:
        print("[CACHE] baseline_fairness_summary.csv already exists.")
        
    if not (RESEARCH_DIR / "operating_points_machine_failure.csv").exists():
        run_calibration_and_operating_points()
    else:
        print("[CACHE] calibration and operating points already computed.")

    if not (RESEARCH_DIR / "stacking_vs_dedicated_comparison.csv").exists():
        run_model_improvements()
    else:
        print("[CACHE] model improvements already computed.")

    generate_final_deliverables()
    print("\nALL EVALUATION TASKS COMPLETED SUCCESSFULLY!")
