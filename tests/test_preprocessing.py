"""
Unit tests for core/preprocessing.py.

No ZenML or MLflow needed — imports from core/ only.
Run with: uv run pytest tests/test_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from core.preprocessing import (
    extract_date_features,
    drop_unwanted_columns,
    build_pipeline,
    prepare_features,
)

SAMPLE_CONFIG = {
    "data": {
        "target_column": "Late_delivery_risk",
        "date_column": "order date (DateOrders)",
    },
    "features": {
        "leakage_columns": ["Delivery Status"],
        "pii_columns": ["Customer Email"],
        "useless_columns": ["Product Description"],
        "id_columns": ["Order Id"],
        "numeric": ["Days for shipment (scheduled)", "Order Item Quantity"],
        "categorical": ["Shipping Mode", "Market"],
    },
}


def make_raw_df(n=50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "order date (DateOrders)": pd.date_range("2016-01-01", periods=n, freq="D").strftime("%m/%d/%Y %H:%M"),
        "Late_delivery_risk": rng.choice([0, 1], size=n),
        "Days for shipment (scheduled)": rng.choice([1, 2, 4], size=n).astype(float),
        "Order Item Quantity": rng.integers(1, 5, size=n).astype(float),
        "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n),
        "Market": rng.choice(["Europe", "LATAM"], size=n),
        "Delivery Status": "Late delivery",  # leakage
        "Customer Email": "x@x.com",          # PII
        "Product Description": None,           # useless
        "Order Id": rng.integers(1000, 9999, size=n),  # ID
    })


class TestExtractDateFeatures:
    def test_creates_four_new_columns(self):
        df = pd.DataFrame({"order date (DateOrders)": ["01/15/2016 10:00"]})
        result = extract_date_features(df, "order date (DateOrders)")
        assert "order_month" in result.columns
        assert "order_day_of_week" in result.columns
        assert "order_is_weekend" in result.columns
        assert "order_quarter" in result.columns

    def test_original_date_column_dropped(self):
        df = pd.DataFrame({"order date (DateOrders)": ["01/15/2016 10:00"]})
        result = extract_date_features(df, "order date (DateOrders)")
        assert "order date (DateOrders)" not in result.columns

    def test_month_value_correct(self):
        df = pd.DataFrame({"order date (DateOrders)": ["06/15/2016 10:00"]})
        result = extract_date_features(df, "order date (DateOrders)")
        assert result["order_month"].iloc[0] == 6.0

    def test_weekend_flag_correct(self):
        # 2016-01-02 is a Saturday
        df = pd.DataFrame({"order date (DateOrders)": ["01/02/2016 10:00"]})
        result = extract_date_features(df, "order date (DateOrders)")
        assert result["order_is_weekend"].iloc[0] == 1.0


class TestDropUnwantedColumns:
    def test_drops_leakage_pii_id_columns(self):
        df = make_raw_df()
        result = drop_unwanted_columns(df, SAMPLE_CONFIG)
        assert "Delivery Status" not in result.columns
        assert "Customer Email" not in result.columns
        assert "Order Id" not in result.columns

    def test_keeps_feature_and_target_columns(self):
        df = make_raw_df()
        result = drop_unwanted_columns(df, SAMPLE_CONFIG)
        assert "Days for shipment (scheduled)" in result.columns
        assert "Shipping Mode" in result.columns
        assert "Late_delivery_risk" in result.columns

    def test_missing_drop_column_does_not_raise(self):
        """Dropping a column that doesn't exist should be silently ignored."""
        df = make_raw_df().drop(columns=["Delivery Status"])
        result = drop_unwanted_columns(df, SAMPLE_CONFIG)  # should not raise
        assert "Late_delivery_risk" in result.columns


class TestBuildPipeline:
    def test_pipeline_has_preprocessor_and_model(self):
        model = LogisticRegression(max_iter=100)
        pipeline = build_pipeline(model, SAMPLE_CONFIG)
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_pipeline_fits_and_predicts(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame({
            "Days for shipment (scheduled)": rng.uniform(1, 5, 100),
            "Order Item Quantity": rng.integers(1, 5, 100).astype(float),
            "Shipping Mode": rng.choice(["Standard Class", "First Class"], 100),
            "Market": rng.choice(["Europe", "LATAM"], 100),
        })
        y = pd.Series(rng.choice([0, 1], 100))
        model = LogisticRegression(max_iter=500)
        pipeline = build_pipeline(model, SAMPLE_CONFIG)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == 100
        assert set(preds).issubset({0, 1})

    def test_unseen_categories_do_not_raise(self):
        """handle_unknown='ignore' means new categories at inference → all-zero encoding."""
        rng = np.random.default_rng(2)
        X_train = pd.DataFrame({
            "Days for shipment (scheduled)": rng.uniform(1, 5, 50),
            "Order Item Quantity": rng.integers(1, 5, 50).astype(float),
            "Shipping Mode": ["Standard Class"] * 50,
            "Market": ["Europe"] * 50,
        })
        X_new = pd.DataFrame({
            "Days for shipment (scheduled)": [3.0],
            "Order Item Quantity": [2.0],
            "Shipping Mode": ["Overnight"],   # unseen category
            "Market": ["Antarctica"],          # unseen category
        })
        y = pd.Series(rng.choice([0, 1], 50))
        model = LogisticRegression(max_iter=500)
        pipeline = build_pipeline(model, SAMPLE_CONFIG)
        pipeline.fit(X_train, y)
        # Should not raise — handle_unknown='ignore' maps unknowns to all-zero vector
        result = pipeline.predict(X_new)
        assert len(result) == 1


class TestPrepareFeatures:
    def test_returns_x_and_y(self):
        df = make_raw_df()
        df = drop_unwanted_columns(df, SAMPLE_CONFIG)
        X, y = prepare_features(df, SAMPLE_CONFIG)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_X(self):
        df = make_raw_df()
        df = drop_unwanted_columns(df, SAMPLE_CONFIG)
        X, y = prepare_features(df, SAMPLE_CONFIG)
        assert "Late_delivery_risk" not in X.columns

    def test_date_features_created(self):
        df = make_raw_df()
        df = drop_unwanted_columns(df, SAMPLE_CONFIG)
        X, _ = prepare_features(df, SAMPLE_CONFIG)
        assert "order_month" in X.columns
        assert "order_quarter" in X.columns
