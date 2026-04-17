"""
Model evaluation step.

Evaluates the trained model on the held-out test set, computes global and
slice-level metrics, checks promotion gates, and logs everything to MLflow.

If promotion gates fail, the step raises — preventing a subpar model from
being registered. Gates are binary: fail = no registration.
"""

import json
import yaml
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from zenml import step

CONFIG_PATH = "configs/training_config.yaml"


@step(experiment_tracker="mlflow_tracker")
def evaluate_model(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_test_raw: pd.DataFrame,
    pipeline: Pipeline,
) -> dict:
    """
    Evaluate model on held-out test set.

    Logs to MLflow:
    - Global metrics: recall, precision, F1, PR-AUC
    - Slice metrics: recall/precision per Shipping Mode, Market, order_month
    - Confusion matrix
    - Promotion gate result

    Args:
        X_test:     preprocessed test features
        y_test:     test labels
        X_test_raw: original (pre-transform) test features for slice grouping
        pipeline:   fully fitted pipeline
        config:     training config dict

    Returns:
        Metrics dict (logged to MLflow, passed to registration step)
    """
    import mlflow
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    from core.evaluation import (
        compute_metrics,
        compute_slice_metrics,
        check_promotion_gates,
        find_best_threshold,
    )

    threshold = config["model"]["prediction_threshold"]
    model = pipeline.named_steps["model"]
    slice_cols = config["evaluation"]["slice_columns"]

    print(f"Evaluating on {len(X_test):,} test samples...")

    # Get probabilities and apply threshold
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Global metrics
    metrics = compute_metrics(y_test, y_pred, y_prob, threshold)
    print(f"\nGlobal metrics (threshold={threshold}):")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  F1:        {metrics['f1']:.3f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.3f}")

    # Suggest optimal threshold (maximise recall subject to precision floor)
    best_thresh = find_best_threshold(
        y_test, y_prob,
        min_precision=config["evaluation"]["min_precision"]
    )
    print(f"\nOptimal threshold (recall↑ s.t. precision≥{config['evaluation']['min_precision']}): {best_thresh}")
    mlflow.log_param("suggested_threshold", best_thresh)

    # Recompute metrics at optimal threshold for comparison
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    metrics_opt = compute_metrics(y_test, y_pred_opt, y_prob, best_thresh)
    print(f"  At optimal threshold — Recall: {metrics_opt['recall']:.3f}, Precision: {metrics_opt['precision']:.3f}")

    # Slice-level metrics (uses raw features for grouping)
    slice_metrics = compute_slice_metrics(
        y_test.reset_index(drop=True),
        pd.Series(y_pred),
        X_test_raw.reset_index(drop=True),
        slice_cols,
        threshold,
    )

    print("\nSlice metrics:")
    for col, col_slices in slice_metrics.items():
        print(f"  {col}:")
        for val, m in sorted(col_slices.items(), key=lambda x: x[1]["recall"]):
            print(f"    {val}: recall={m['recall']:.3f}, precision={m['precision']:.3f}, n={m['n_samples']:,}")

    # Log all metrics to MLflow
    mlflow.log_metric("test_recall", metrics["recall"])
    mlflow.log_metric("test_precision", metrics["precision"])
    mlflow.log_metric("test_f1", metrics["f1"])
    mlflow.log_metric("test_pr_auc", metrics["pr_auc"])
    mlflow.log_metric("test_samples", metrics["n_samples"])
    mlflow.log_metric("optimal_recall", metrics_opt["recall"])
    mlflow.log_metric("optimal_precision", metrics_opt["precision"])

    # Log slice metrics
    for col, col_slices in slice_metrics.items():
        for val, m in col_slices.items():
            safe_val = val.replace(" ", "_").replace("/", "_")
            mlflow.log_metric(f"slice_{col[:8]}_{safe_val[:10]}_recall", m["recall"])
            mlflow.log_metric(f"slice_{col[:8]}_{safe_val[:10]}_precision", m["precision"])

    # Log confusion matrix as MLflow artifact
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax,
                                            display_labels=["On Time", "Late"])
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    plt.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close()

    # Log PR curve
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax2,
                                            name="LightGBM")
    ax2.axhline(y=config["evaluation"]["min_precision"], color="red",
                linestyle="--", label=f"Min precision={config['evaluation']['min_precision']}")
    ax2.axvline(x=config["evaluation"]["min_recall"], color="orange",
                linestyle="--", label=f"Min recall={config['evaluation']['min_recall']}")
    ax2.legend()
    ax2.set_title("Precision-Recall Curve")
    plt.tight_layout()
    mlflow.log_figure(fig2, "pr_curve.png")
    plt.close()

    # Check promotion gates
    passed, failures = check_promotion_gates(metrics_opt, slice_metrics, config)

    if passed:
        print("\n✓ All promotion gates passed — model eligible for registration")
        mlflow.log_param("gates_passed", True)
    else:
        mlflow.log_param("gates_passed", False)
        mlflow.log_param("gate_failures", json.dumps(failures))
        print(f"\n✗ Promotion gates FAILED:")
        for f in failures:
            print(f"    - {f}")
        raise ValueError(
            f"Model did not pass promotion gates:\n" +
            "\n".join(f"  - {f}" for f in failures)
        )

    return {
        "metrics": metrics,
        "metrics_optimal_threshold": metrics_opt,
        "slice_metrics": slice_metrics,
        "threshold": threshold,
        "optimal_threshold": best_thresh,
    }
