"""
Tests for core/evaluation.py.

All tests use synthetic data — no ZenML, no MLflow, no CSV needed.
"""

import numpy as np
import pandas as pd
import pytest

from core.evaluation import (
    compute_metrics,
    compute_slice_metrics,
    check_promotion_gates,
    find_best_threshold,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_predictions():
    y_true = pd.Series([1, 1, 1, 0, 0, 0])
    y_pred = pd.Series([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.85, 0.8, 0.1, 0.15, 0.2])
    return y_true, y_pred, y_prob


@pytest.fixture
def mixed_predictions():
    # 6 positives, 4 negatives; model catches 4/6 positives (recall=0.667)
    y_true = pd.Series([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])  # 2 FN, 1 FP
    y_prob = np.array([0.9, 0.8, 0.75, 0.65, 0.4, 0.35, 0.2, 0.15, 0.55, 0.1])
    return y_true, y_pred, y_prob


@pytest.fixture
def config():
    return {
        "evaluation": {
            "min_recall": 0.70,
            "min_precision": 0.65,
            "min_slice_recall": 0.35,
        }
    }


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def test_compute_metrics_returns_required_keys(perfect_predictions):
    y_true, y_pred, y_prob = perfect_predictions
    result = compute_metrics(y_true, y_pred, y_prob)
    for key in ("recall", "precision", "f1", "pr_auc", "threshold", "n_samples", "n_positives"):
        assert key in result, f"Missing key: {key}"


def test_compute_metrics_perfect_scores(perfect_predictions):
    y_true, y_pred, y_prob = perfect_predictions
    result = compute_metrics(y_true, y_pred, y_prob)
    assert result["recall"] == 1.0
    assert result["precision"] == 1.0
    assert result["f1"] == 1.0


def test_compute_metrics_counts(mixed_predictions):
    y_true, y_pred, y_prob = mixed_predictions
    result = compute_metrics(y_true, y_pred, y_prob)
    assert result["n_samples"] == 10
    assert result["n_positives"] == 6


def test_compute_metrics_recall_value(mixed_predictions):
    y_true, y_pred, y_prob = mixed_predictions
    result = compute_metrics(y_true, y_pred, y_prob)
    # 4 true positives out of 6 actual positives
    assert abs(result["recall"] - 4 / 6) < 0.01


def test_compute_metrics_threshold_stored(mixed_predictions):
    y_true, y_pred, y_prob = mixed_predictions
    result = compute_metrics(y_true, y_pred, y_prob, threshold=0.42)
    assert result["threshold"] == 0.42


def test_compute_metrics_all_negative_predictions():
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([0, 0, 0, 0])
    y_prob = np.array([0.4, 0.3, 0.2, 0.1])
    result = compute_metrics(y_true, y_pred, y_prob)
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0


# ---------------------------------------------------------------------------
# compute_slice_metrics
# ---------------------------------------------------------------------------

def test_compute_slice_metrics_structure():
    y_true = pd.Series([1, 1, 0, 0, 1, 0] * 4)
    y_pred = pd.Series([1, 0, 0, 0, 1, 0] * 4)
    X_test = pd.DataFrame({
        "Shipping Mode": ["First Class"] * 12 + ["Standard Class"] * 12,
    })
    result = compute_slice_metrics(y_true, y_pred, X_test, ["Shipping Mode"])

    assert "Shipping Mode" in result
    for val in result["Shipping Mode"].values():
        assert "recall" in val
        assert "precision" in val
        assert "n_samples" in val


def test_compute_slice_metrics_skips_missing_columns():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0, 0])
    X_test = pd.DataFrame({"Shipping Mode": ["A", "A", "B", "B"]})
    # Request a column that doesn't exist — should not raise
    result = compute_slice_metrics(y_true, y_pred, X_test, ["Shipping Mode", "Market"])
    assert "Shipping Mode" in result
    assert "Market" not in result


def test_compute_slice_metrics_skips_small_slices():
    # Only 3 samples in one slice — below the 10-sample minimum
    y_true = pd.Series([1, 0, 1] + [1, 0] * 10)
    y_pred = pd.Series([1, 0, 0] + [1, 0] * 10)
    X_test = pd.DataFrame({
        "Region": ["Small"] * 3 + ["Large"] * 20
    })
    result = compute_slice_metrics(y_true, y_pred, X_test, ["Region"])
    assert "Small" not in result.get("Region", {})
    assert "Large" in result.get("Region", {})


# ---------------------------------------------------------------------------
# check_promotion_gates
# ---------------------------------------------------------------------------

def test_promotion_gates_pass(config):
    metrics = {"recall": 0.75, "precision": 0.70}
    slice_metrics = {
        "Shipping Mode": {"First Class": {"recall": 0.90, "precision": 0.85, "n_samples": 100}}
    }
    passed, failures = check_promotion_gates(metrics, slice_metrics, config)
    assert passed is True
    assert failures == []


def test_promotion_gates_fail_recall(config):
    metrics = {"recall": 0.60, "precision": 0.70}  # below 0.70
    passed, failures = check_promotion_gates(metrics, {}, config)
    assert passed is False
    assert any("Recall" in f for f in failures)


def test_promotion_gates_fail_precision(config):
    metrics = {"recall": 0.80, "precision": 0.50}  # below 0.65
    passed, failures = check_promotion_gates(metrics, {}, config)
    assert passed is False
    assert any("Precision" in f for f in failures)


def test_promotion_gates_fail_slice_recall(config):
    metrics = {"recall": 0.80, "precision": 0.70}
    slice_metrics = {
        "Shipping Mode": {"Standard Class": {"recall": 0.20, "precision": 0.50, "n_samples": 100}}
    }
    passed, failures = check_promotion_gates(metrics, slice_metrics, config)
    assert passed is False
    assert any("Standard Class" in f for f in failures)


def test_promotion_gates_multiple_failures(config):
    metrics = {"recall": 0.50, "precision": 0.40}
    passed, failures = check_promotion_gates(metrics, {}, config)
    assert passed is False
    assert len(failures) == 2


# ---------------------------------------------------------------------------
# find_best_threshold
# ---------------------------------------------------------------------------

def test_find_best_threshold_returns_float():
    y_true = pd.Series([1, 1, 0, 0, 1, 0] * 20)
    y_prob = np.array([0.8, 0.75, 0.2, 0.15, 0.7, 0.1] * 20)
    t = find_best_threshold(y_true, y_prob, min_precision=0.65)
    assert isinstance(t, float)
    assert 0.10 <= t <= 0.90


def test_find_best_threshold_meets_precision_floor():
    from sklearn.metrics import precision_score
    y_true = pd.Series([1, 1, 1, 0, 0, 0] * 30)
    y_prob = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1] * 30)
    min_prec = 0.65
    t = find_best_threshold(y_true, y_prob, min_precision=min_prec)
    y_pred = (y_prob >= t).astype(int)
    actual_precision = precision_score(y_true, y_pred, zero_division=0)
    assert actual_precision >= min_prec - 0.01  # allow tiny floating point slack
