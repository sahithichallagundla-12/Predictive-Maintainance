"""Generate evaluation graphs, confusion matrices, and ROC/PR curves for all six targets."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

ROOT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = ROOT_DIR / "artifacts" / "training" / "reports"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "training" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("machine_failure", "Machine Failure (Overall)"),
    ("twf", "TWF (Tool Wear Failure)"),
    ("hdf", "HDF (Heat Dissipation Failure)"),
    ("pwf", "PWF (Power Failure)"),
    ("osf", "OSF (Overstrain Failure)"),
    ("rnf", "RNF (Random Failure)"),
]

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.sans-serif": "Arial", "font.family": "sans-serif"})

# 1. Plot Confusion Matrices (2x3 Grid)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (target_key, target_title) in enumerate(TARGETS):
    ax = axes[idx]
    pred_file = REPORTS_DIR / f"{target_key}_test_predictions.csv"
    report_file = REPORTS_DIR / f"{target_key}_test_report.json"
    
    if not pred_file.exists():
        ax.set_title(f"{target_title} (Not Found)")
        continue
        
    df = pd.read_csv(pred_file)
    cm = confusion_matrix(df["actual"], df["prediction"], labels=[0, 1])
    
    threshold = 0.5
    if report_file.exists():
        with open(report_file) as rf:
            rep = json.load(rf)
            threshold = rep.get("metrics", {}).get("threshold", 0.5)

    # Display heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{target_title}\n(Threshold = {threshold:.2f})", fontsize=12, fontweight="bold", pad=8)
    
    # Tick marks
    classes = ["Negative (0)", "Positive (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    
    # Print numbers with contrast
    thresh_val = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            label_text = f"{val:,}\n({val/cm.sum()*100:.1f}%)"
            ax.text(
                j, i, label_text,
                ha="center", va="center",
                color="white" if val > thresh_val else "black",
                fontsize=11, fontweight="bold"
            )
            
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Confusion Matrices on Held-out Test Set (N = 1,500)", fontsize=16, fontweight="bold", y=0.99)
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrices.png"
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {cm_path}")

# 2. Plot ROC and PR Curves (1x2 Grid)
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for (target_key, target_title), color in zip(TARGETS, colors):
    pred_file = REPORTS_DIR / f"{target_key}_test_predictions.csv"
    if not pred_file.exists():
        continue
    df = pd.read_csv(pred_file)
    y_true = df["actual"]
    y_prob = df["probability"]
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{target_title} (AUC = {roc_auc:.3f})")
    
    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    ax_pr.plot(recall, precision, color=color, lw=2, label=f"{target_title} (AP = {pr_auc:.3f})")

# ROC styling
ax_roc.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=12, fontweight="bold")
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
ax_roc.legend(loc="lower right", fontsize=9, frameon=True)
ax_roc.grid(True, linestyle="--", alpha=0.5)

# PR styling
ax_pr.set_xlim([0.0, 1.0])
ax_pr.set_ylim([0.0, 1.05])
ax_pr.set_xlabel("Recall", fontsize=12, fontweight="bold")
ax_pr.set_ylabel("Precision", fontsize=12, fontweight="bold")
ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=14, fontweight="bold")
ax_pr.legend(loc="upper right", fontsize=9, frameon=True)
ax_pr.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Model Discriminative Performance on Held-Out Test Set", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
roc_pr_path = OUTPUT_DIR / "roc_pr_curves.png"
plt.savefig(roc_pr_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {roc_pr_path}")

# 3. Plot Metrics Comparison Bar Chart
metrics_data = []
for target_key, target_title in TARGETS:
    report_file = REPORTS_DIR / f"{target_key}_test_report.json"
    if report_file.exists():
        with open(report_file) as rf:
            rep = json.load(rf)
            m = rep.get("metrics", {})
            metrics_data.append({
                "Target": target_title.split(" (")[0],
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "F1 Score": m.get("f1", 0),
                "Accuracy": m.get("accuracy", 0),
            })

df_metrics = pd.DataFrame(metrics_data)
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(df_metrics))
width = 0.25

rects1 = ax.bar(x - width, df_metrics["Precision"], width, label="Precision", color="#3498db")
rects2 = ax.bar(x, df_metrics["Recall"], width, label="Recall", color="#2ecc71")
rects3 = ax.bar(x + width, df_metrics["F1 Score"], width, label="F1 Score", color="#e74c3c")

ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title("Test Set Performance by Failure Target (Precision, Recall, F1)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(df_metrics["Target"], fontsize=11, fontweight="bold")
ax.set_ylim([0, 1.15])
ax.legend(fontsize=11)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
metrics_path = OUTPUT_DIR / "metrics_comparison.png"
plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {metrics_path}")
