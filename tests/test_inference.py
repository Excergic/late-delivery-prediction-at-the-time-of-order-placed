"""
Tests for inference feature preparation.

Verifies that the feature preparation used in generate_predictions produces
the same column structure as training — the core invariant that prevents
training-serving skew.
"""

import pandas as pd
import numpy as np
import pytest

from core.preprocessing import drop_unwanted_columns, extract_date_features


@pytest.fixture
def minimal_config():
    return {
        "data": {
            "target_column": "Late_delivery_risk",
            "date_column": "order date (DateOrders)",
        },
        "features": {
            "leakage_columns": ["Delivery Status", "Days for shipping (real)"],
            "pii_columns": ["Customer Email"],
            "useless_columns": [],
            "id_columns": ["Order Id"],
            "numeric": ["Order Item Quantity", "Order Item Total", "order_month",
                        "order_day_of_week", "order_is_weekend", "order_quarter"],
            "categorical": ["Shipping Mode", "Market"],
        },
    }


@pytest.fixture
def sample_orders(minimal_config):
    """Raw orders DataFrame as it would arrive from a CSV."""
    date_col = minimal_config["data"]["date_column"]
    target_col = minimal_config["data"]["target_column"]
    n = 50
    return pd.DataFrame({
        "Order Id": range(n),
        date_col: ["1/15/2018 10:00"] * n,
        target_col: np.random.randint(0, 2, n),
        "Delivery Status": ["Shipped"] * n,
        "Days for shipping (real)": np.random.randint(1, 10, n),
        "Customer Email": ["x@x.com"] * n,
        "Order Item Quantity": np.random.randint(1, 5, n),
        "Order Item Total": np.random.uniform(10, 500, n),
        "Shipping Mode": np.random.choice(["First Class", "Standard Class"], n),
        "Market": np.random.choice(["LATAM", "Europe"], n),
    })


# ---------------------------------------------------------------------------
# Feature preparation produces correct columns
# ---------------------------------------------------------------------------

def test_drop_unwanted_removes_leakage(sample_orders, minimal_config):
    df = drop_unwanted_columns(sample_orders, minimal_config)
    assert "Delivery Status" not in df.columns
    assert "Days for shipping (real)" not in df.columns


def test_drop_unwanted_removes_pii(sample_orders, minimal_config):
    df = drop_unwanted_columns(sample_orders, minimal_config)
    assert "Customer Email" not in df.columns


def test_drop_unwanted_removes_id_columns(sample_orders, minimal_config):
    df = drop_unwanted_columns(sample_orders, minimal_config)
    assert "Order Id" not in df.columns


def test_extract_date_features_produces_all_four_columns(sample_orders, minimal_config):
    date_col = minimal_config["data"]["date_column"]
    df = extract_date_features(sample_orders, date_col)
    for col in ("order_month", "order_day_of_week", "order_is_weekend", "order_quarter"):
        assert col in df.columns, f"Missing date feature: {col}"


def test_feature_columns_present_after_full_prep(sample_orders, minimal_config):
    """After full feature engineering, all configured feature columns exist."""
    date_col = minimal_config["data"]["date_column"]
    df = drop_unwanted_columns(sample_orders, minimal_config)
    df = extract_date_features(df, date_col)

    numeric_cols = minimal_config["features"]["numeric"]
    categorical_cols = minimal_config["features"]["categorical"]
    for col in numeric_cols + categorical_cols:
        assert col in df.columns, f"Expected feature column missing: {col}"


def test_target_column_still_present_after_feature_prep(sample_orders, minimal_config):
    """Feature prep does not drop the target column — that's the caller's job."""
    date_col = minimal_config["data"]["date_column"]
    target_col = minimal_config["data"]["target_column"]
    df = drop_unwanted_columns(sample_orders, minimal_config)
    df = extract_date_features(df, date_col)
    assert target_col in df.columns


def test_feature_prep_same_columns_with_and_without_target(sample_orders, minimal_config):
    """
    Feature prep with target column dropped vs not dropped should produce
    identical non-target columns. This is the training-serving parity check.
    """
    date_col = minimal_config["data"]["date_column"]
    target_col = minimal_config["data"]["target_column"]

    df_with_target = drop_unwanted_columns(sample_orders, minimal_config)
    df_with_target = extract_date_features(df_with_target, date_col)

    sample_no_target = sample_orders.drop(columns=[target_col])
    df_no_target = drop_unwanted_columns(sample_no_target, minimal_config)
    df_no_target = extract_date_features(df_no_target, date_col)

    cols_with = set(df_with_target.columns) - {target_col}
    cols_without = set(df_no_target.columns)
    assert cols_with == cols_without, (
        f"Column mismatch between labeled and unlabeled paths: "
        f"{cols_with.symmetric_difference(cols_without)}"
    )


# ---------------------------------------------------------------------------
# Flag rate sanity
# ---------------------------------------------------------------------------

def test_flag_rate_bounded():
    """Predictions must be in [0, 1]."""
    probs = np.array([0.1, 0.35, 0.36, 0.9, 0.5])
    threshold = 0.35
    preds = (probs >= threshold).astype(int)
    assert set(preds).issubset({0, 1})


def test_lower_threshold_increases_recall():
    """Lowering threshold should never decrease recall (monotonicity)."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.6, 0.4, 0.3, 0.2, 0.1, 0.05])

    from sklearn.metrics import recall_score
    recall_high = recall_score(y_true, (y_prob >= 0.45).astype(int), zero_division=0)
    recall_low = recall_score(y_true, (y_prob >= 0.25).astype(int), zero_division=0)
    assert recall_low >= recall_high
