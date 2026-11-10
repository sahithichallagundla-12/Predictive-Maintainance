"""Train and evaluate AI4I 2020 predictive-maintenance classifiers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


TARGET_COLUMNS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
CATEGORICAL_COLUMNS = ["Type"]
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED_COLUMNS = [
    "Temperature difference [K]",
    "Mechanical power [W]",
    "Torque-wear interaction",
]
NUMERIC_COLUMNS = SENSOR_COLUMNS + ENGINEERED_COLUMNS
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with domain-informed features derived from sensor values."""
    features = frame.copy()
    features["Temperature difference [K]"] = (
        features["Process temperature [K]"] - features["Air temperature [K]"]
    )
    features["Mechanical power [W]"] = (
        features["Torque [Nm]"] * 2.0 * np.pi * features["Rotational speed [rpm]"] / 60.0
    )
    features["Torque-wear interaction"] = (
        features["Torque [Nm]"] * features["Tool wear [min]"]
    )
    return features


def load_dataset(data_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not data_path.is_file():
        raise FileNotFoundError("Dataset not found: {}".format(data_path))

    frame = pd.read_csv(data_path)
    required = set(IDENTIFIER_COLUMNS + CATEGORICAL_COLUMNS + SENSOR_COLUMNS + TARGET_COLUMNS)
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError("Dataset is missing required columns: {}".format(missing_columns))

    original_rows = len(frame)
    frame = frame.drop_duplicates().copy()
    for column in SENSOR_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for target in TARGET_COLUMNS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        if frame[target].isna().any():
            raise ValueError("Target column {!r} contains missing or non-numeric values.".format(target))
        observed_values = set(frame[target].unique())
        if not observed_values.issubset({0, 1}):
            raise ValueError(
                "Target column {!r} must contain only 0 and 1; got {}.".format(
                    target, sorted(observed_values)
                )
            )
        frame[target] = frame[target].astype(int)

    if frame.empty:
        raise ValueError("Dataset contains no rows after duplicate removal.")
    if frame[IDENTIFIER_COLUMNS[0]].isna().any():
        raise ValueError("UDI contains missing values and cannot identify prediction rows.")

    counts = {
        "rows_before_deduplication": int(original_rows),
        "rows_after_deduplication": int(len(frame)),
        "duplicates_removed": int(original_rows - len(frame)),
        "missing_sensor_values": int(frame[SENSOR_COLUMNS].isna().sum().sum()),
    }
    return frame, counts


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), NUMERIC_COLUMNS),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def build_models(
    y_train: np.ndarray, random_state: int
) -> Dict[str, Tuple[Any, str, bool]]:
    positive = int(np.sum(y_train == 1))
    negative = int(np.sum(y_train == 0))
    if positive == 0 or negative == 0:
        raise ValueError("Training data must contain both classes for every target.")

    return {
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "tree",
            False,
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            "tree",
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=negative / positive,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=random_state,
                verbosity=0,
            ),
            "tree",
            False,
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
                verbosity=-1,
            ),
            "tree",
            False,
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=False,
                random_seed=random_state,
                allow_writing_files=False,
            ),
            "tree",
            True,
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=random_state,
            ),
            "tree",
            True,
        ),
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=random_state,
            ),
            "scaled",
            False,
        ),
    }


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    candidates = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        candidates.append(
            (
                float(f1_score(y_true, predictions, zero_division=0)),
                float(recall_score(y_true, predictions, zero_division=0)),
                float(threshold),
            )
        )
    best_f1, _, best_threshold = max(candidates, key=lambda result: (result[0], result[1], -result[2]))
    return best_threshold, calculate_metrics(y_true, probabilities, best_threshold)


def calculate_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else None,
        "pr_auc": float(average_precision_score(y_true, probabilities)) if has_both_classes else None,
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def split_dataset(
    frame: pd.DataFrame, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    labels = frame["Machine failure"].to_numpy()
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=labels,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=random_state,
        stratify=labels[temporary_indices],
    )
    return train_indices, validation_indices, test_indices


def clean_target_name(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", target).strip("_").lower()


def run_training(data_path: Path, output_dir: Path, random_state: int) -> Dict[str, Any]:
    frame, data_quality = load_dataset(data_path)
    features = engineer_features(frame)
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    X = features[feature_columns]

    train_indices, validation_indices, test_indices = split_dataset(frame, random_state)
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    preprocessor_tree = make_preprocessor(scale_numeric=False)
    preprocessor_scaled = make_preprocessor(scale_numeric=True)
    X_train_tree = preprocessor_tree.fit_transform(X.iloc[train_indices])
    X_validation_tree = preprocessor_tree.transform(X.iloc[validation_indices])
    X_test_tree = preprocessor_tree.transform(X.iloc[test_indices])
    X_train_scaled = preprocessor_scaled.fit_transform(X.iloc[train_indices])
    X_validation_scaled = preprocessor_scaled.transform(X.iloc[validation_indices])
    X_test_scaled = preprocessor_scaled.transform(X.iloc[test_indices])
    preprocessors = {"tree": preprocessor_tree, "scaled": preprocessor_scaled}
    matrices = {
        "tree": (X_train_tree, X_validation_tree, X_test_tree),
        "scaled": (X_train_scaled, X_validation_scaled, X_test_scaled),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_summary: Dict[str, Dict[str, Any]] = {}
    for split_name, indices in split_indices.items():
        split_summary[split_name] = {
            "rows": int(len(indices)),
            "targets": {
                target: {
                    "positive": int(frame.iloc[indices][target].sum()),
                    "negative": int(len(indices) - frame.iloc[indices][target].sum()),
                }
                for target in TARGET_COLUMNS
            },
        }

    comparison_rows: List[Dict[str, Any]] = []
    best_models_summary: Dict[str, Any] = {}

    for target in TARGET_COLUMNS:
        y = frame[target].to_numpy()
        y_train = y[train_indices]
        y_validation = y[validation_indices]
        y_test = y[test_indices]
        models = build_models(y_train, random_state)
        successful: List[Dict[str, Any]] = []

        print("\nTraining target: {}".format(target))
        for model_name, (estimator, feature_kind, use_sample_weight) in models.items():
            X_train, X_validation, _ = matrices[feature_kind]
            try:
                fit_options: Dict[str, Any] = {}
                if use_sample_weight:
                    fit_options["sample_weight"] = compute_sample_weight("balanced", y_train)
                estimator.fit(X_train, y_train, **fit_options)
                validation_probabilities = estimator.predict_proba(X_validation)[:, 1]
                threshold, validation_metrics = select_threshold(
                    y_validation, validation_probabilities
                )
                result = {
                    "target": target,
                    "model": model_name,
                    "feature_processing": feature_kind,
                    "status": "success",
                    "validation_threshold": threshold,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_precision": validation_metrics["precision"],
                    "validation_recall": validation_metrics["recall"],
                    "validation_f1": validation_metrics["f1"],
                    "validation_roc_auc": validation_metrics["roc_auc"],
                    "validation_pr_auc": validation_metrics["pr_auc"],
                    "error": "",
                    "estimator": estimator,
                }
                successful.append(result)
                print(
                    "  {:22s} val F1={:.4f} recall={:.4f} precision={:.4f} threshold={:.2f}".format(
                        model_name,
                        validation_metrics["f1"],
                        validation_metrics["recall"],
                        validation_metrics["precision"],
                        threshold,
                    )
                )
            except Exception as exc:
                comparison_rows.append(
                    {
                        "target": target,
                        "model": model_name,
                        "feature_processing": feature_kind,
                        "status": "failed",
                        "validation_threshold": None,
                        "validation_accuracy": None,
                        "validation_precision": None,
                        "validation_recall": None,
                        "validation_f1": None,
                        "validation_roc_auc": None,
                        "validation_pr_auc": None,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
                print("  {} failed for {}: {}".format(model_name, target, exc), file=sys.stderr)

        if not successful:
            raise RuntimeError("No model trained successfully for target {!r}.".format(target))

        best = max(
            successful,
            key=lambda result: (
                result["validation_f1"],
                result["validation_recall"],
                -result["validation_threshold"],
            ),
        )
        best_estimator = best["estimator"]
        best_kind = best["feature_processing"]
        _, _, X_test = matrices[best_kind]
        test_probabilities = best_estimator.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, test_probabilities, best["validation_threshold"])
        test_predictions = (test_probabilities >= best["validation_threshold"]).astype(int)
        report = classification_report(
            y_test, test_predictions, labels=[0, 1], zero_division=0, output_dict=True
        )

        bundle = {
            "estimator": best_estimator,
            "preprocessor": preprocessors[best_kind],
            "target": target,
            "model_name": best["model"],
            "feature_processing": best_kind,
            "threshold": float(best["validation_threshold"]),
            "input_feature_columns": feature_columns,
            "engineered_feature_columns": ENGINEERED_COLUMNS,
            "identifier_columns": IDENTIFIER_COLUMNS,
            "random_state": int(random_state),
        }
        safe_target = clean_target_name(target)
        joblib.dump(bundle, models_dir / "best_{}.joblib".format(safe_target))

        with (reports_dir / "{}_test_report.json".format(safe_target)).open(
            "w", encoding="utf-8"
        ) as report_file:
            json.dump(
                {"metrics": test_metrics, "classification_report": report},
                report_file,
                indent=2,
            )

        pd.DataFrame(
            {
                "UDI": frame.iloc[test_indices]["UDI"].to_numpy(),
                "actual": y_test,
                "probability": test_probabilities,
                "prediction": test_predictions,
                "threshold": float(best["validation_threshold"]),
            }
        ).to_csv(reports_dir / "{}_test_predictions.csv".format(safe_target), index=False)

        for result in successful:
            comparison_rows.append(
                {key: value for key, value in result.items() if key != "estimator"}
            )

        best_models_summary[target] = {
            "best_model": best["model"],
            "feature_processing": best_kind,
            "validation_selected_threshold": float(best["validation_threshold"]),
            "validation_metrics": {
                "accuracy": best["validation_accuracy"],
                "precision": best["validation_precision"],
                "recall": best["validation_recall"],
                "f1": best["validation_f1"],
                "roc_auc": best["validation_roc_auc"],
                "pr_auc": best["validation_pr_auc"],
            },
            "test_metrics": test_metrics,
            "test_classification_report": report,
        }
        print(
            "  BEST {} | test F1={:.4f} recall={:.4f} precision={:.4f}".format(
                best["model"], test_metrics["f1"], test_metrics["recall"], test_metrics["precision"]
            )
        )

    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "model_validation_comparison.csv", index=False
    )
    with (output_dir / "best_models_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(best_models_summary, summary_file, indent=2)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "dataset": str(data_path.resolve()),
                "random_state": int(random_state),
                "data_quality": data_quality,
                "split_percentages": {"train": 70, "validation": 15, "test": 15},
                "split_summary": split_summary,
                "features": feature_columns,
                "targets": TARGET_COLUMNS,
                "models_compared": list(build_models(frame[TARGET_COLUMNS[0]].iloc[train_indices].to_numpy(), random_state)),
                "threshold_selection": "Maximum validation F1; ties favor recall then the lower threshold.",
                "test_set_used_for_model_selection": False,
                "warning": (
                    "AI4I 2020 is synthetic. RNF has very few positive examples, "
                    "so its metrics can vary substantially."
                ),
            },
            metadata_file,
            indent=2,
        )

    print("\nTraining and evaluation completed.")
    print("Outputs saved in {}".format(output_dir.resolve()))
    print(json.dumps(best_models_summary, indent=2))
    return best_models_summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_dir / "ai4i2020.csv",
        help="Path to the AI4I CSV dataset (default: ai4i2020.csv beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "artifacts" / "training",
        help="Directory for trained models and evaluation reports.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splits and models.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_training(args.data, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
