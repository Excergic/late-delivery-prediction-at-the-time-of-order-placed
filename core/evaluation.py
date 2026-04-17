"""
Model evaluation logic.

All metric computation lives here — no ZenML, no MLflow.
The ZenML step calls these functions and handles logging to MLflow.

Covers:
- Global metrics (recall, precision, F1, PR-AUC)
- Slice-level metrics (per Shipping Mode, Market, order_month)
- Promotion gate checks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics for a binary classifier.

    Args:
        y_true:    true labels (0/1)
        y_pred:    predicted labels at the given threshold
        y_prob:    predicted probabilities for the positive class
        threshold: decision threshold used to produce y_pred

    Returns:
        dict with recall, precision, f1, pr_auc, threshold, confusion_matrix
    """
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "pr_auc": round(float(pr_auc), 4),
        "threshold": threshold,
        "confusion_matrix": cm,
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
        "positive_rate": round(float(y_true.mean()), 4),
    }


def compute_slice_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    X_test: pd.DataFrame,
    slice_columns: list[str],
    threshold: float = 0.5,
) -> dict:
    """
    Compute recall and precision broken down by each slice column.

    A model that achieves Recall=0.82 overall but Recall=0.55 for
    Southeast Asia shipments is not production-ready. Slice metrics catch this.

    Returns:
        Nested dict: {column_name: {slice_value: {recall, precision, n_samples}}}
    """
    slices = {}
    for col in slice_columns:
        if col not in X_test.columns:
            continue
        slices[col] = {}
        for val in X_test[col].unique():
            mask = X_test[col] == val
            if mask.sum() < 10:
                continue  # skip slices too small to be meaningful
            slices[col][str(val)] = {
                "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "precision": round(float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "n_samples": int(mask.sum()),
            }
    return slices


def check_promotion_gates(
    metrics: dict,
    slice_metrics: dict,
    config: dict,
) -> tuple[bool, list[str]]:
    """
    Check whether a model passes all promotion gates.
    Gates are binary — fail = no promotion.

    Returns:
        (passed: bool, failures: list of failure messages)
    """
    failures = []
    eval_cfg = config["evaluation"]

    # Global gates
    if metrics["recall"] < eval_cfg["min_recall"]:
        failures.append(
            f"Recall {metrics['recall']:.3f} < required {eval_cfg['min_recall']:.3f}"
        )
    if metrics["precision"] < eval_cfg["min_precision"]:
        failures.append(
            f"Precision {metrics['precision']:.3f} < required {eval_cfg['min_precision']:.3f}"
        )

    # Slice-level gates
    min_slice = eval_cfg["min_slice_recall"]
    for col, col_slices in slice_metrics.items():
        for val, val_metrics in col_slices.items():
            if val_metrics["recall"] < min_slice:
                failures.append(
                    f"Slice [{col}={val}] recall {val_metrics['recall']:.3f} < required {min_slice:.3f}"
                )

    return len(failures) == 0, failures


def find_best_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    min_precision: float = 0.65,
) -> float:
    """
    Find the lowest threshold that still meets the precision floor.
    Lower threshold = higher recall. We want maximum recall subject to
    precision >= min_precision.

    Returns:
        Best threshold (float between 0 and 1)
    """
    best_threshold = 0.5
    best_recall = 0.0

    for t in np.arange(0.10, 0.90, 0.01):
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_threshold = t

    return round(float(best_threshold), 2)
