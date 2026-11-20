"""Export preprocessed datasets using the same logic as train.py.

This script reuses load_dataset(), engineer_features(), and make_preprocessor()
from train.py WITHOUT modifying any of that existing code.

It produces two CSV files in preprocessed_data/:
  - preprocessed_tree.csv   : median-imputed + one-hot encoded (for tree models)
  - preprocessed_scaled.csv : same but also StandardScaled (for MLP / linear models)

Both files include the identifier columns (UDI, Product ID), all target columns,
and a 'split' column indicating train / validation / test membership.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Import everything directly from train.py (no changes to that file) ────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (
    CATEGORICAL_COLUMNS,
    ENGINEERED_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMNS,
    engineer_features,
    load_dataset,
    make_preprocessor,
    split_dataset,
)

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent
DATA_PATH     = PROJECT_DIR / "ai4i2020.csv"
OUTPUT_DIR    = PROJECT_DIR / "preprocessed_data"
RANDOM_STATE  = 42          # same seed as train.py default


def get_ohe_feature_names(preprocessor) -> list[str]:
    """Return the column names produced by the fitted ColumnTransformer."""
    ohe = preprocessor.named_transformers_["categorical"]["onehot"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_COLUMNS))
    return NUMERIC_COLUMNS + cat_names


def export() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load (same as train.py) ─────────────────────────────────────────
    print("Loading dataset from:", DATA_PATH)
    frame, counts = load_dataset(DATA_PATH)
    print(f"  Rows after dedup: {counts['rows_after_deduplication']}")
    print(f"  Missing sensor values: {counts['missing_sensor_values']}")

    # ── 2. Feature engineering (same as train.py) ──────────────────────────
    features = engineer_features(frame)
    feature_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    X = features[feature_columns]

    # ── 3. Train/val/test split (same seed as train.py) ───────────────────
    train_idx, val_idx, test_idx = split_dataset(frame, RANDOM_STATE)
    split_labels = np.empty(len(frame), dtype=object)
    split_labels[train_idx] = "train"
    split_labels[val_idx]   = "validation"
    split_labels[test_idx]  = "test"

    # ── 4. Fit preprocessors on TRAIN split only, then transform all rows ──
    for variant, scale in [("tree", False), ("scaled", True)]:
        preprocessor = make_preprocessor(scale_numeric=scale)

        # Fit on training rows only (same as train.py)
        preprocessor.fit(X.iloc[train_idx])

        # Transform the full dataset
        X_all = preprocessor.transform(X)

        # Reconstruct column names
        col_names = get_ohe_feature_names(preprocessor)

        # Build output DataFrame
        df_out = pd.DataFrame(X_all, columns=col_names, index=frame.index)

        # Prepend identifiers, targets, and split label
        for col in reversed(IDENTIFIER_COLUMNS):
            df_out.insert(0, col, frame[col].values)
        for col in TARGET_COLUMNS:
            df_out[col] = frame[col].values
        df_out["split"] = split_labels

        out_path = OUTPUT_DIR / f"preprocessed_{variant}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\n[{variant}] Saved → {out_path}")
        print(f"  Shape : {df_out.shape}")
        print(f"  Columns: {list(df_out.columns)}")

    print("\nDone. Check the 'preprocessed_data/' folder.")


if __name__ == "__main__":
    export()

