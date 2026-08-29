"""Evaluation helpers: threshold selection, metrics, and diagnostic plots.

Churn is imbalanced (~26.5% positive), so the headline metrics are PR-AUC and recall
at a business-chosen threshold rather than accuracy.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5


def choose_threshold(y_true, y_proba) -> float:
    """Probability cut-off that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more point than thresholds; drop the last.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def score_at_threshold(y_true, y_proba, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Threshold-free ranking metrics plus point metrics at ``threshold``."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }


def evaluate_model(pipeline, x_test, y_test, threshold: float | None = None) -> dict:
    """Fit-free evaluation of an already-fitted pipeline on the held-out set."""
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    if threshold is None:
        threshold = choose_threshold(y_test, y_proba)
    return score_at_threshold(y_test, y_proba, threshold)


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Turn ``{model_name: metrics_dict}`` into a sorted comparison table."""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)


def plot_pr_curve(pipeline, x_test, y_test, ax=None):
    """Precision-recall curve for one fitted pipeline."""
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ax = ax or plt.gca()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve (AP = {average_precision_score(y_test, y_proba):.3f})")
    return ax


def plot_confusion(pipeline, x_test, y_test, threshold: float, ax=None):
    """Confusion matrix at the chosen operating threshold."""
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    ax = ax or plt.gca()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix @ {threshold:.2f}")
    return ax
