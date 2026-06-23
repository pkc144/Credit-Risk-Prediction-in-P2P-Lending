"""Model evaluation: metrics, confusion matrix, ROC curve, feature importance.

Each plot is written under reports/figures/ in a category-specific subfolder
so it can be regenerated without overwriting earlier runs of other models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

OUTPUT_ROOT = Path("reports/figures")


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

    def as_row(self) -> dict[str, float | str]:
        return {
            "Model": self.model_name,
            "Accuracy": round(self.accuracy, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
            "AUC": round(self.auc, 4),
        }


def compute_metrics(y_true, y_pred, y_prob, model_name: str) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, y_prob),
    )


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: Path = OUTPUT_ROOT / "confusion_matrices") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        annot_kws={"size": 14},
        xticklabels=["0", "1"],
        yticklabels=["0", "1"],
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = out_dir / f"{model_name.lower()}_cm.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_roc(y_true, y_prob, model_name: str, out_dir: Path = OUTPUT_ROOT / "auc_curves") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = out_dir / f"{model_name.lower()}_roc.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_top_features(
    importances: Iterable[float],
    feature_names: Iterable[str],
    model_name: str,
    n: int = 15,
    out_dir: Path = OUTPUT_ROOT / "top_15_features",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    s = pd.Series(list(importances), index=list(feature_names)).sort_values(ascending=True).tail(n)
    plt.figure(figsize=(8, 6))
    s.plot(kind="barh", color="steelblue")
    plt.title(f"Top {n} Features — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    path = out_dir / f"{model_name.lower()}_top{n}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def save_metrics_table(results: list[EvalResult], out_dir: Path = OUTPUT_ROOT / "evaluation_metrics") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.as_row() for r in results])
    csv_path = out_dir / "evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(df) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.4)
    plt.title("Model Evaluation Metrics", fontsize=14, pad=12)
    img_path = out_dir / "evaluation_metrics.png"
    plt.savefig(img_path, dpi=200, bbox_inches="tight")
    plt.close()
    return csv_path


def report(y_true, y_pred, y_prob, model_name: str, feature_names=None, importances=None) -> EvalResult:
    res = compute_metrics(y_true, y_pred, y_prob, model_name)
    print(
        f"[{model_name}] acc={res.accuracy:.4f} prec={res.precision:.4f} "
        f"rec={res.recall:.4f} f1={res.f1:.4f} auc={res.auc:.4f}"
    )
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_roc(y_true, y_prob, model_name)
    if importances is not None and feature_names is not None:
        plot_top_features(importances, feature_names, model_name)
    return res
