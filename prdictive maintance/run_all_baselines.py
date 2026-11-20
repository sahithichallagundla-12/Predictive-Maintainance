"""run_all_baselines.py: Parameterized runner for all published literature baselines on AI4I 2020.

Baselines re-implemented:
1. B1: Matzka (2020) - Bagged Trees (DecisionTreeClassifier ensemble)
2. B2: Tuama et al. (2025) - Random Forest & XGBoost
3. B3: Waghulde et al. (2025) - Random Forest, XGBoost, SVM (RBF), DNN (MLP 64-32)
4. Reference floor: Logistic Regression and Always-Negative classifier

Supports:
- --seed <int>: seed for split and model random_state
- --features <raw|engineered>: test fairness (whether baseline has access to engineered features)
- --output <path>: save CSV of test results
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from train import (
    load_dataset,
    split_dataset,
    engineer_features,
    SENSOR_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMNS,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "ai4i2020.csv"


def build_pipeline_data(frame, seed, use_engineered=True):
    train_idx, val_idx, test_idx = split_dataset(frame, seed)
    if use_engineered:
        feat_df = engineer_features(frame)
        num_cols = NUMERIC_COLUMNS
    else:
        feat_df = frame.copy()
        num_cols = SENSOR_COLUMNS

    feature_cols = num_cols + CATEGORICAL_COLUMNS
    X = feat_df[feature_cols]

    # Preprocessors
    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )
    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ]
    )

    X_train_unscaled = preprocessor_unscaled.fit_transform(X.iloc[train_idx])
    X_test_unscaled = preprocessor_unscaled.transform(X.iloc[test_idx])

    X_train_scaled = preprocessor_scaled.fit_transform(X.iloc[train_idx])
    X_test_scaled = preprocessor_scaled.transform(X.iloc[test_idx])

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "unscaled": (X_train_unscaled, X_test_unscaled),
        "scaled": (X_train_scaled, X_test_scaled),
    }


def evaluate_model(name, family, source, y_train, y_test, X_train, X_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    has_both = len(np.unique(y_test)) == 2
    roc = roc_auc_score(y_test, y_prob) if has_both else np.nan
    pr = average_precision_score(y_test, y_prob) if has_both else np.nan

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "source": source,
        "model": name,
        "family": family,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc,
        "pr_auc": pr,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "test_pos": int(y_test.sum())
    }


def run_baselines_for_seed(seed=42, use_engineered=False):
    frame, _ = load_dataset(DATA_PATH)
    splits = build_pipeline_data(frame, seed, use_engineered=use_engineered)
    X_tr_u, X_te_u = splits["unscaled"]
    X_tr_s, X_te_s = splits["scaled"]
    train_idx = splits["train_idx"]
    test_idx = splits["test_idx"]

    results = []
    for target in TARGET_COLUMNS:
        y_train = frame[target].iloc[train_idx].to_numpy()
        y_test = frame[target].iloc[test_idx].to_numpy()

        models = [
            # Floor
            ("Logistic Regression", "linear", "Reference Floor", True,
             LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
            # B1 Matzka 2020
            ("Bagged Trees", "ensemble", "B1 Matzka 2020", False,
             BaggingClassifier(estimator=DecisionTreeClassifier(random_state=seed), n_estimators=100, random_state=seed)),
            # B2 / B3 Tree Models
            ("Random Forest", "tree", "B2 Tuama / B3 Waghulde", False,
             RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=1)),
            ("XGBoost", "boosting", "B2 Tuama / B3 Waghulde", False,
             XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, eval_metric="logloss", n_jobs=1)),
            # B3 Scaled Models
            ("SVM (RBF)", "kernel", "B3 Waghulde 2025", True,
             SVC(probability=True, class_weight="balanced", random_state=seed)),
            ("DNN 64-32", "neural", "B3 Waghulde 2025", True,
             MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed, early_stopping=True)),
        ]

        for name, fam, src, scaled, mdl in models:
            X_tr = X_tr_s if scaled else X_tr_u
            X_te = X_te_s if scaled else X_te_u
            res = evaluate_model(name, fam, src, y_train, y_test, X_tr, X_te, mdl)
            res["target"] = target
            res["seed"] = seed
            res["features"] = "engineered" if use_engineered else "raw"
            results.append(res)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", choices=["raw", "engineered"], default="raw")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    use_eng = (args.features == "engineered")
    df = run_baselines_for_seed(seed=args.seed, use_engineered=use_eng)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} baseline rows to {out_path}")
    else:
        print(df[["target", "model", "f1", "recall", "pr_auc"]].to_string())


if __name__ == "__main__":
    main()
