"""
Final Model Evaluation Script for Aegis Predictive Maintenance
==============================================================
- Reuses existing test predictions and trained model artifacts
- Does NOT retrain models or tune thresholds on test data
- Produces all required outputs in artifacts/training/final_evaluation/
"""

from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
REPORTS_DIR  = ROOT / "artifacts" / "training" / "reports"
MODELS_DIR   = ROOT / "artifacts" / "training" / "models"
VAL_CSV      = ROOT / "artifacts" / "training" / "model_validation_comparison.csv"
SUMMARY_JSON = ROOT / "artifacts" / "training" / "best_models_summary.json"
OUT_DIR      = ROOT / "artifacts" / "training" / "final_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_KEYS = {
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
ALGO_DISPLAY = {
    "RandomForest":         "Random Forest",
    "ExtraTrees":           "Extra Trees (Bagging)",
    "XGBoost":              "XGBoost",
    "LightGBM":             "LightGBM",
    "CatBoost":             "CatBoost",
    "HistGradientBoosting": "Hist Gradient Boosting",
    "MLP":                  "MLP / DNN",
}
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading existing test predictions …")
test_data: dict[str, pd.DataFrame] = {}
for target, key in TARGET_KEYS.items():
    p = REPORTS_DIR / f"{key}_test_predictions.csv"
    test_data[target] = pd.read_csv(p)

with open(SUMMARY_JSON) as fh:
    best_summary = json.load(fh)

val_df = pd.read_csv(VAL_CSV)


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE FINAL TEST METRICS FOR ALL 6 TARGETS
# ══════════════════════════════════════════════════════════════════════════════
print("Computing final test metrics …")

def compute_metrics(y_true, y_prob, threshold, model_name, feature_proc):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    roc_auc = roc_auc_score(y_true, y_prob) if has_both else float("nan")
    pr_auc  = average_precision_score(y_true, y_prob) if has_both else float("nan")
    return {
        "model":        model_name,
        "feature_proc": feature_proc,
        "threshold":    round(threshold, 4),
        "accuracy":     round(accuracy_score(y_true, y_pred), 6),
        "precision":    round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall":       round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1":           round(f1_score(y_true, y_pred, zero_division=0), 6),
        "roc_auc":      round(roc_auc, 6) if not np.isnan(roc_auc) else "N/A",
        "pr_auc":       round(pr_auc,  6) if not np.isnan(pr_auc)  else "N/A",
        "specificity":  round(specificity, 6),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

metrics_rows = []
for target in TARGETS:
    df  = test_data[target]
    info = best_summary[target]
    m = compute_metrics(
        df["actual"].values, df["probability"].values,
        info["validation_selected_threshold"],
        info["best_model"], info["feature_processing"],
    )
    m["target"] = target
    metrics_rows.append(m)
    print(f"  {target:20s} model={info['best_model']:22s} "
          f"F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']}")

metrics_df = pd.DataFrame(metrics_rows)[[
    "target","model","feature_proc","threshold",
    "accuracy","precision","recall","f1",
    "roc_auc","pr_auc","specificity",
    "tn","fp","fn","tp",
]]
metrics_df.to_csv(OUT_DIR / "final_metrics_all_targets.csv", index=False)
print("  Saved → final_metrics_all_targets.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALGORITHM COMPARISON (Machine failure, all 7 algorithms)
#    Uses validation metrics from model_validation_comparison.csv (no test leakage)
#    plus test metrics for the best (selected) model only.
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding algorithm comparison …")

mf_val = val_df[val_df["target"] == "Machine failure"].copy()
best_model_name = best_summary["Machine failure"]["best_model"]

algo_rows = []
for _, row in mf_val.iterrows():
    algo_rows.append({
        "algorithm":           ALGO_DISPLAY.get(row["model"], row["model"]),
        "feature_processing":  row["feature_processing"],
        "val_threshold":       round(row["validation_threshold"], 4),
        "val_accuracy":        round(row["validation_accuracy"], 6),
        "val_precision":       round(row["validation_precision"], 6),
        "val_recall":          round(row["validation_recall"], 6),
        "val_f1":              round(row["validation_f1"], 6),
        "val_roc_auc":         round(row["validation_roc_auc"], 6),
        "val_pr_auc":          round(row["validation_pr_auc"], 6),
        "selected_as_best":    row["model"] == best_model_name,
        "note": (
            "Best model — test metrics available in final_metrics_all_targets.csv"
            if row["model"] == best_model_name
            else "Validation only (threshold tuned on val set; not evaluated on test)"
        ),
    })

algo_df = pd.DataFrame(algo_rows).sort_values("val_f1", ascending=False)
algo_df.to_csv(OUT_DIR / "algorithm_comparison.csv", index=False)
print("  Saved → algorithm_comparison.csv")
print(f"\n  Algorithm comparison (Machine failure, validation set):")
for _, r in algo_df.iterrows():
    star = " ★" if r["selected_as_best"] else ""
    print(f"    {r['algorithm']:30s} val_F1={r['val_f1']:.4f}  val_ROC-AUC={r['val_roc_auc']:.4f}{star}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots …")

# ── 4a. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes = axes.flatten()
for idx, target in enumerate(TARGETS):
    ax  = axes[idx]
    row = metrics_df[metrics_df["target"] == target].iloc[0]
    cm  = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{target}\n(Model: {row['model']} | τ={row['threshold']:.2f})",
                 fontsize=11, fontweight="bold", pad=6)
    classes = ["Negative (0)", "Positive (1)"]
    ticks   = np.arange(2)
    ax.set_xticks(ticks); ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual",    fontsize=10, fontweight="bold")
    thresh_val = cm.max() / 2.0
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}\n({v/total*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if v > thresh_val else "black",
                    fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Confusion Matrices — Held-out Test Set (N=1,500)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → confusion_matrices.png")

# ── 4b. ROC Curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for idx, target in enumerate(TARGETS):
    df = test_data[target]
    y_true, y_prob = df["actual"].values, df["probability"].values
    if len(np.unique(y_true)) < 2:
        continue
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val      = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, lw=2, color=COLORS[idx], label=f"{target} (AUC={auc_val:.4f})")

ax.plot([0,1],[0,1], "k--", lw=1.2, label="Random chance")
ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
ax.set_ylabel("True Positive Rate",  fontsize=12, fontweight="bold")
ax.set_title("ROC Curves — Held-out Test Set", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → roc_curves.png")

# ── 4c. Precision-Recall Curves ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for idx, target in enumerate(TARGETS):
    df = test_data[target]
    y_true, y_prob = df["actual"].values, df["probability"].values
    if len(np.unique(y_true)) < 2:
        continue
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(rec, prec, lw=2, color=COLORS[idx], label=f"{target} (AP={ap:.4f})")

ax.set_xlabel("Recall",    fontsize=12, fontweight="bold")
ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
ax.set_title("Precision-Recall Curves — Held-out Test Set", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "precision_recall_curves.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → precision_recall_curves.png")

# ── 4d. Metrics Comparison Bar Chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(TARGETS))
width = 0.22
metrics_to_plot = ["precision", "recall", "f1", "roc_auc"]
bar_colors      = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
bar_labels      = ["Precision", "Recall", "F1", "ROC-AUC"]
offsets         = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

for m, color, label, offset in zip(metrics_to_plot, bar_colors, bar_labels, offsets):
    vals = []
    for target in TARGETS:
        v = metrics_df[metrics_df["target"]==target].iloc[0][m]
        vals.append(float(v) if v != "N/A" else 0.0)
    bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.annotate(f"{v:.2f}",
                        xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(TARGETS, fontsize=11, fontweight="bold")
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.18)
ax.set_title("Final Test-Set Performance by Failure Target", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(OUT_DIR / "metrics_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved → metrics_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXCEL WORKBOOK
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Excel workbook …")

try:
    import openpyxl
    from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side,
                                  numbers)
    from openpyxl.utils import get_column_letter
    from openpyxl.chart import BarChart, Reference, LineChart
    from openpyxl.chart.series import SeriesLabel

    HEX_HEADER  = "1F4E79"
    HEX_SUBHEAD = "2E75B6"
    HEX_BEST    = "E2EFDA"
    HEX_WARN    = "FFF2CC"
    HEX_DANGER  = "FFE0E0"

    def header_cell(ws, row, col, value, bold=True, bg=HEX_HEADER, fg="FFFFFF", size=11):
        c = ws.cell(row=row, column=col, value=value)
        c.font      = Font(bold=bold, color=fg, size=size)
        c.fill      = PatternFill("solid", fgColor=bg)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border    = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin"))
        return c

    def data_cell(ws, row, col, value, num_format=None, bg=None, bold=False):
        c = ws.cell(row=row, column=col, value=value)
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border    = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin"))
        if num_format: c.number_format = num_format
        if bg:   c.fill = PatternFill("solid", fgColor=bg)
        if bold: c.font = Font(bold=True)
        return c

    def autofit(ws, extra=4):
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + extra, 40)

    wb = openpyxl.Workbook()

    # ── Sheet 1: Overall Performance ─────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Overall Performance"
    ws1.row_dimensions[1].height = 36
    ws1.merge_cells("A1:O1")
    t = ws1.cell(row=1, column=1, value="AEGIS PREDICTIVE MAINTENANCE — FINAL MODEL EVALUATION")
    t.font      = Font(bold=True, size=16, color="1F4E79")
    t.alignment = Alignment(horizontal="center", vertical="center")

    ws1.merge_cells("A2:O2")
    sub = ws1.cell(row=2, column=1,
                   value="Dataset: AI4I 2020 (Synthetic) | Test Set: 1,500 rows (15% stratified) | Threshold tuned on validation set only")
    sub.font      = Font(italic=True, size=10, color="595959")
    sub.alignment = Alignment(horizontal="center")

    headers = ["Target","Best Model","Feature Proc","Threshold",
               "Accuracy","Precision","Recall","F1",
               "ROC-AUC","PR-AUC","Specificity",
               "TN","FP","FN","TP"]
    for ci, h in enumerate(headers, 1):
        header_cell(ws1, 3, ci, h, bg=HEX_HEADER)

    for ri, row in enumerate(metrics_df.itertuples(index=False), 4):
        vals = list(row)
        f1_val = float(vals[7]) if vals[7] != "N/A" else 0.0
        roc_val = float(vals[8]) if vals[8] != "N/A" else 0.0
        bg = HEX_BEST if f1_val >= 0.85 else (HEX_WARN if f1_val >= 0.5 else HEX_DANGER)
        for ci, v in enumerate(vals, 1):
            fmt = "0.0000" if isinstance(v, float) else None
            data_cell(ws1, ri, ci, v, num_format=fmt, bg=bg)

    # Notes row
    note_row = 4 + len(metrics_df)
    ws1.merge_cells(f"A{note_row}:O{note_row}")
    n = ws1.cell(row=note_row, column=1,
                 value="⚠ HDF/PWF perfect scores (1.0) reflect very few test positives (13-18 cases) in synthetic data — not guaranteed real-world reliability. "
                       "TWF/RNF show F1=0.0 on test despite non-zero validation metrics — extremely rare classes (2-8 test positives).")
    n.font      = Font(italic=True, size=9, color="843C0C")
    n.alignment = Alignment(wrap_text=True)
    ws1.row_dimensions[note_row].height = 42
    autofit(ws1)

    # ── Sheet 2: Six Target Evaluation ───────────────────────────────────────
    ws2 = wb.create_sheet("Six Target Evaluation")
    ws2.row_dimensions[1].height = 30
    ws2.merge_cells("A1:H1")
    t2 = ws2.cell(row=1, column=1, value="Per-Target Evaluation Details — Best Model per Failure Type")
    t2.font      = Font(bold=True, size=14, color="1F4E79")
    t2.alignment = Alignment(horizontal="center", vertical="center")

    current_row = 3
    detail_headers = ["Metric","Value","Interpretation"]
    for target in TARGETS:
        row_data = metrics_df[metrics_df["target"] == target].iloc[0]
        ws2.merge_cells(f"A{current_row}:H{current_row}")
        tc = ws2.cell(row=current_row, column=1, value=f"▶ {target}")
        tc.font      = Font(bold=True, size=12, color="FFFFFF")
        tc.fill      = PatternFill("solid", fgColor=HEX_SUBHEAD)
        tc.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws2.row_dimensions[current_row].height = 22
        current_row += 1

        for ci, h in enumerate(detail_headers, 1):
            header_cell(ws2, current_row, ci, h, bg="2E75B6", size=10)
        for ci in range(4, 9):
            ws2.cell(row=current_row, column=ci).value = ""
        current_row += 1

        f1_v = float(row_data["f1"]) if row_data["f1"] != "N/A" else 0
        roc_v = float(row_data["roc_auc"]) if row_data["roc_auc"] != "N/A" else 0
        details = [
            ("Selected Model",   row_data["model"],       "Best by validation F1"),
            ("Threshold (τ)",    row_data["threshold"],   "Tuned on validation set — not test"),
            ("Accuracy",         row_data["accuracy"],    f"{'High' if row_data['accuracy']>0.98 else 'Moderate'} — caution: class imbalance inflates accuracy"),
            ("Precision",        row_data["precision"],   f"{'Good' if float(row_data['precision'])>0.8 else 'Poor — many FPs'}" if row_data["precision"] != "N/A" else ""),
            ("Recall",           row_data["recall"],      f"{'Good' if float(row_data['recall'])>0.8 else 'Poor — misses failures'}" if row_data["recall"] != "N/A" else ""),
            ("F1 Score",         row_data["f1"],          f"{'Reliable' if f1_v>=0.8 else 'Moderate' if f1_v>=0.5 else 'Unreliable — too few positives'}"),
            ("ROC-AUC",          row_data["roc_auc"],     f"{'Excellent (≥0.95)' if roc_v>=0.95 else 'Good (≥0.85)' if roc_v>=0.85 else 'Poor'}"),
            ("PR-AUC",           row_data["pr_auc"],      "More informative than ROC for imbalanced classes"),
            ("Specificity (TNR)",row_data["specificity"], "True Negative Rate — ability to avoid false alarms"),
            ("Confusion Matrix", f"TN={row_data['tn']} FP={row_data['fp']} FN={row_data['fn']} TP={row_data['tp']}", ""),
            ("Test Positives",   int(row_data["tp"]) + int(row_data["fn"]), "Actual failure cases in test set"),
        ]
        for metric_name, val, interp in details:
            data_cell(ws2, current_row, 1, metric_name, bold=True)
            data_cell(ws2, current_row, 2, val, num_format="0.0000" if isinstance(val, float) else None)
            c3 = ws2.cell(row=current_row, column=3, value=interp)
            c3.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            current_row += 1

        current_row += 1

    for col in ws2.columns:
        ws2.column_dimensions[get_column_letter(col[0].column)].width = 30

    # ── Sheet 3: Algorithm Comparison ────────────────────────────────────────
    ws3 = wb.create_sheet("Algorithm Comparison")
    ws3.row_dimensions[1].height = 30
    ws3.merge_cells("A1:L1")
    t3 = ws3.cell(row=1, column=1,
                  value="Algorithm Comparison — Machine Failure Target | Validation Set Metrics (threshold tuned on val; test not used)")
    t3.font      = Font(bold=True, size=13, color="1F4E79")
    t3.alignment = Alignment(horizontal="center", vertical="center")

    algo_headers = ["Algorithm","Feature Proc","Val Threshold","Val Accuracy",
                    "Val Precision","Val Recall","Val F1","Val ROC-AUC","Val PR-AUC","Selected as Best","Note"]
    for ci, h in enumerate(algo_headers, 1):
        header_cell(ws3, 2, ci, h, bg=HEX_HEADER)

    for ri, row in enumerate(algo_df.itertuples(index=False), 3):
        is_best = row.selected_as_best
        bg = HEX_BEST if is_best else None
        vals = [row.algorithm, row.feature_processing, row.val_threshold,
                row.val_accuracy, row.val_precision, row.val_recall,
                row.val_f1, row.val_roc_auc, row.val_pr_auc,
                "★ BEST" if is_best else "", row.note]
        for ci, v in enumerate(vals, 1):
            fmt = "0.0000" if isinstance(v, float) else None
            data_cell(ws3, ri, ci, v, num_format=fmt, bg=bg, bold=is_best)

    note_r = 3 + len(algo_df)
    ws3.merge_cells(f"A{note_r}:K{note_r}")
    nr = ws3.cell(row=note_r, column=1,
                  value="IMPORTANT: All metrics above are from the validation set. "
                        "Only the selected best model (HistGradientBoosting) has test-set metrics — see 'Overall Performance' sheet. "
                        "Other algorithms are compared on validation only to avoid test-set leakage.")
    nr.font      = Font(italic=True, size=9, color="843C0C")
    nr.alignment = Alignment(wrap_text=True)
    ws3.row_dimensions[note_r].height = 36
    autofit(ws3)

    # ── Sheet 4: Confusion Matrix Summary ────────────────────────────────────
    ws4 = wb.create_sheet("Confusion Matrix")
    ws4.row_dimensions[1].height = 28
    ws4.merge_cells("A1:I1")
    t4 = ws4.cell(row=1, column=1, value="Confusion Matrix Summary — All Six Targets — Test Set (N=1,500)")
    t4.font      = Font(bold=True, size=14, color="1F4E79")
    t4.alignment = Alignment(horizontal="center", vertical="center")

    cm_headers = ["Target","Model","Threshold",
                  "TN (correct negatives)","FP (false alarms)",
                  "FN (missed failures)","TP (caught failures)",
                  "Total Test Positives","Miss Rate (FN/TP+FN)"]
    for ci, h in enumerate(cm_headers, 1):
        header_cell(ws4, 2, ci, h, bg=HEX_HEADER)

    for ri, row in enumerate(metrics_df.itertuples(index=False), 3):
        total_pos = row.tp + row.fn
        miss_rate = row.fn / total_pos if total_pos > 0 else float("nan")
        bg = HEX_DANGER if miss_rate > 0.5 or total_pos <= 2 else (HEX_WARN if miss_rate > 0.2 else HEX_BEST)
        vals = [row.target, row.model, row.threshold,
                row.tn, row.fp, row.fn, row.tp, total_pos,
                f"{miss_rate:.1%}" if not np.isnan(miss_rate) else "N/A"]
        for ci, v in enumerate(vals, 1):
            data_cell(ws4, ri, ci, v, bg=bg)

    autofit(ws4)

    # ── Sheet 5: Failure Mode Analysis ───────────────────────────────────────
    ws5 = wb.create_sheet("Failure Mode Analysis")
    ws5.row_dimensions[1].height = 28
    ws5.merge_cells("A1:F1")
    t5 = ws5.cell(row=1, column=1, value="Failure Mode Analysis — Reliability, Class Imbalance & Recommendations")
    t5.font      = Font(bold=True, size=14, color="1F4E79")
    t5.alignment = Alignment(horizontal="center", vertical="center")

    fa_headers = ["Failure Mode","Test Positives","Reliability","Key Issue","Strength","Recommendation"]
    for ci, h in enumerate(fa_headers, 1):
        header_cell(ws5, 2, ci, h, bg=HEX_HEADER)

    fa_data = [
        ("Machine failure","51","✅ High — F1=0.85, ROC-AUC=0.989",
         "7 false alarms, 8 missed failures",
         "Best overall; HistGradBoost excellent",
         "Production-candidate; validate threshold on real data"),
        ("HDF","18","✅ Perfect — F1=1.0 (synthetic pattern)",
         "Very few test cases; may not generalise",
         "Clear deterministic pattern in AI4I",
         "Treat with caution; re-evaluate on real data"),
        ("PWF","13","✅ Perfect — F1=1.0 (synthetic pattern)",
         "Only 13 test positives",
         "Clear power-failure signature",
         "Same caution as HDF"),
        ("OSF","15","✅ Strong — F1=0.94, ROC-AUC=1.0",
         "2 false alarms; zero missed",
         "Perfect recall — catches all failures",
         "Monitor precision; tune threshold if FP cost high"),
        ("TWF","8","❌ Poor — F1=0.0 on test",
         "Only 8 test positives; threshold mismatch",
         "ROC-AUC=0.946 — model scores are ordered",
         "Gather more data; consider lower threshold or cost-sensitive learning"),
        ("RNF","2","❌ Unpredictable — F1=0.0 on test",
         "Only 2 test positives — essentially unevaluable",
         "By definition random; no reliable feature signal",
         "Remove from production scope; flag as 'unknown' in dashboard"),
    ]
    tier_colors = {
        "✅ High":    "E2EFDA",
        "✅ Perfect": "FFF2CC",
        "✅ Strong":  "DEEAF1",
        "❌ Poor":    "FFE0E0",
        "❌ Unpredictable": "FFD7D7",
    }
    for ri, row in enumerate(fa_data, 3):
        bg = next((v for k, v in tier_colors.items() if k in row[2]), None)
        for ci, v in enumerate(row, 1):
            c = data_cell(ws5, ri, ci, v, bg=bg)
            c.alignment = Alignment(horizontal="left", vertical="center",
                                     wrap_text=True, indent=1)
        ws5.row_dimensions[ri].height = 42

    for col, width in zip("ABCDEF", [22,16,30,32,30,42]):
        ws5.column_dimensions[col].width = width

    wb.save(OUT_DIR / "Aegis_Final_Model_Evaluation.xlsx")
    print("  Saved → Aegis_Final_Model_Evaluation.xlsx")

except ImportError:
    print("  openpyxl not available — skipping Excel. Install with: pip install openpyxl")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\nWriting final evaluation report …")

def fmt(v):
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

report_lines = [
    "# Aegis Predictive Maintenance — Final Model Evaluation Report",
    "",
    "> **Dataset:** AI4I 2020 Predictive Maintenance (Synthetic) — 10,000 rows  ",
    "> **Test set:** 1,500 rows (15%, stratified on Machine failure, seed=42)  ",
    "> **Threshold selection:** Validation set only (1,500 rows, 15%) — test set never used for tuning  ",
    "> **Date generated:** auto-generated by run_final_evaluation.py  ",
    "",
    "---",
    "",
    "## 1. Final Test-Set Metrics — All Six Targets",
    "",
    "| Target | Model | τ | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Specificity | TN | FP | FN | TP |",
    "|--------|-------|---|----------|-----------|--------|----|---------|--------|-------------|----|----|----|----|",
]
for _, row in metrics_df.iterrows():
    report_lines.append(
        f"| {row['target']} | {row['model']} | {fmt(row['threshold'])} "
        f"| {fmt(row['accuracy'])} | {fmt(row['precision'])} | {fmt(row['recall'])} "
        f"| {fmt(row['f1'])} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} "
        f"| {fmt(row['specificity'])} | {row['tn']} | {row['fp']} | {row['fn']} | {row['tp']} |"
    )

report_lines += [
    "",
    "> ⚠ **Important caveats:**",
    "> - HDF and PWF show perfect scores (1.0) due to clear synthetic patterns with very few test positives (13–18 cases).",
    "> - TWF (8 test positives) and RNF (2 test positives) show F1=0.0 on the test set — the classes are too rare to evaluate reliably.",
    "> - All metrics are on the held-out test set. The threshold τ was selected on the validation set.",
    "",
    "---",
    "",
    "## 2. Algorithm Comparison — Machine Failure Target (Validation Set)",
    "",
    "| Algorithm | Val τ | Val Precision | Val Recall | Val F1 | Val ROC-AUC | Selected |",
    "|-----------|-------|--------------|------------|--------|-------------|----------|",
]
for _, row in algo_df.iterrows():
    star = "★" if row["selected_as_best"] else ""
    report_lines.append(
        f"| {row['algorithm']} | {fmt(row['val_threshold'])} "
        f"| {fmt(row['val_precision'])} | {fmt(row['val_recall'])} "
        f"| {fmt(row['val_f1'])} | {fmt(row['val_roc_auc'])} | {star} |"
    )

report_lines += [
    "",
    "> **Note:** Comparison metrics above are from the validation set for all algorithms.",
    "> Only the selected model (HistGradientBoosting) is evaluated on the test set.",
    "> Test F1 for Machine failure = **0.8515**, Test ROC-AUC = **0.9891**.",
    "",
    "---",
    "",
    "## 3. System Evaluation",
    "",
    "### 3.1 Overall Machine Failure Prediction",
    "The **HistGradientBoosting** classifier achieves the strongest balance on the Machine failure target:",
    "- Test F1 = **0.8515** at threshold τ = 0.65",
    "- ROC-AUC = **0.9891** — excellent discrimination",
    "- 43 of 51 actual failures detected (recall 84.3%)",
    "- Only 7 false alarms out of 1,449 true negatives (specificity 99.5%)",
    "",
    "### 3.2 Individual Failure-Mode Prediction",
    "| Mode | Reliability | Key Finding |",
    "|------|-------------|-------------|",
    "| HDF  | ✅ Excellent | Perfect test scores; likely a deterministic synthetic rule |",
    "| PWF  | ✅ Excellent | Same as HDF — very clear signal in AI4I data |",
    "| OSF  | ✅ Strong    | F1=0.94, 0 missed failures, 2 false alarms |",
    "| Machine failure | ✅ Good | F1=0.85 — best practically meaningful result |",
    "| TWF  | ❌ Poor     | F1=0.0 on test; only 8 test positives — too rare |",
    "| RNF  | ❌ Unreliable | F1=0.0 on test; only 2 positives — unevaluable |",
    "",
    "### 3.3 Class Imbalance & Rare Failures",
    "- The dataset is heavily imbalanced: Machine failure rate ≈ 3.4%, RNF ≈ 0.03%",
    "- Accuracy is a misleading metric here (a naive 'always predict 0' gives 96.6% accuracy)",
    "- F1 and PR-AUC are the most informative metrics for imbalanced targets",
    "- TWF and RNF have so few test positives that no threshold can produce reliable F1",
    "- Class weights and `scale_pos_weight` mitigate but cannot overcome near-zero prevalence",
    "",
    "### 3.4 Strengths",
    "- Pipeline correctly prioritises recall (catching failures) while limiting false alarms",
    "- Threshold tuned on validation — test results are unbiased",
    "- 7 diverse algorithms compared; best selected by validation F1",
    "- SHAP explanations provide interpretable, feature-level insight",
    "- RAG-based diagnosis grounded in maintenance manual evidence",
    "",
    "### 3.5 Limitations",
    "- **Synthetic data only** — AI4I 2020 does not represent any specific real asset or sensor setup",
    "- Perfect scores for HDF/PWF are likely artefacts of synthetic rules, not generalizable performance",
    "- TWF and RNF cannot be reliably evaluated with available data volume",
    "- No temporal validation — the train/val/test split is random, not time-based",
    "- No calibration check — predicted probabilities may not be well-calibrated",
    "",
    "---",
    "",
    "## 4. Final Conclusion",
    "",
    "**Best model for Machine failure:** `HistGradientBoosting` (validation F1=0.880, test F1=0.852, test ROC-AUC=0.989)",
    "",
    "**Reliably predicted failure categories:** HDF, PWF, OSF (strong synthetic patterns), Machine failure (best practical result)",
    "",
    "**Categories needing improvement:** TWF and RNF — insufficient positive examples in both training and test.",
    "These require substantially more failure data before any model can be evaluated reliably.",
    "",
    "**Why HistGradientBoosting is appropriate:**",
    "- Highest validation F1 among all 7 candidates",
    "- Native handling of missing values (median imputation still applied for consistency)",
    "- Gradient boosting captures the non-linear interactions between temperature, torque, and wear",
    "- Fast inference suitable for near-real-time dashboard use",
    "",
    "**Dataset limitations:**",
    "- Results on AI4I 2020 are not transferable to real equipment without re-training on real sensor data",
    "- Perfect scores for some targets are synthetic artefacts",
    "- This system is a **research prototype**, not a production safety system",
    "",
    "---",
    "",
    "## 5. Generated Output Files",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| `final_metrics_all_targets.csv` | Full test-set metrics for all 6 targets |",
    "| `algorithm_comparison.csv` | Validation metrics for all 7 algorithms on Machine failure |",
    "| `confusion_matrices.png` | 2×3 grid of confusion matrices |",
    "| `roc_curves.png` | ROC curves for all 6 targets |",
    "| `precision_recall_curves.png` | PR curves for all 6 targets |",
    "| `metrics_comparison.png` | Grouped bar chart of Precision/Recall/F1/ROC-AUC |",
    "| `Aegis_Final_Model_Evaluation.xlsx` | Full Excel workbook (5 sheets) |",
    "| `final_evaluation_report.md` | This report |",
    "",
    "All files saved to: `artifacts/training/final_evaluation/`",
]

with open(OUT_DIR / "final_evaluation_report.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))
print("  Saved → final_evaluation_report.md")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY TABLE IN TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print("  FINAL EVALUATION COMPLETE")
print("="*90)
print(f"\n  Output directory: {OUT_DIR}\n")
print(f"  {'Target':<20} {'Model':<24} {'τ':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC-AUC':>9} {'PR-AUC':>8}")
print(f"  {'-'*95}")
for _, row in metrics_df.iterrows():
    print(f"  {row['target']:<20} {row['model']:<24} {fmt(row['threshold']):>5} "
          f"{fmt(row['accuracy']):>7} {fmt(row['precision']):>7} {fmt(row['recall']):>7} "
          f"{fmt(row['f1']):>7} {fmt(row['roc_auc']):>9} {fmt(row['pr_auc']):>8}")
print(f"\n  Files generated:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"    ✓ {f.name}  ({f.stat().st_size:,} bytes)")
print()

