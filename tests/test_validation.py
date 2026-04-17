"""
Unit tests for core/validation.py.

These tests import from core/ only — no ZenML, no MLflow needed.
Run with: uv run pytest tests/test_validation.py -v
"""

import pytest
import pandas as pd
import numpy as np

from core.validation import validate_raw_data, validate_feature_matrix


# Minimal config matching training_config.yaml structure
SAMPLE_CONFIG = {
    "data": {
        "target_column": "Late_delivery_risk",
        "min_rows": 10,
        "max_rows": 500_000,
        "target_min_rate": 0.40,
        "target_max_rate": 0.70,
    },
    "features": {
        "leakage_columns": ["Delivery Status", "Days for shipping (real)"],
        "numeric": ["Days for shipment (scheduled)", "Order Item Quantity"],
        "categorical": ["Shipping Mode", "Market"],
    },
}


def make_valid_df(n=100) -> pd.DataFrame:
    """Create a minimal valid DataFrame that passes all gates."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Late_delivery_risk": rng.choice([0, 1], size=n, p=[0.45, 0.55]),
        "Days for shipment (scheduled)": rng.integers(1, 7, size=n).astype(float),
        "Order Item Quantity": rng.integers(1, 10, size=n).astype(float),
        "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n),
        "Market": rng.choice(["Europe", "Pacific Asia"], size=n),
    })


class TestValidateRawData:
    def test_valid_data_passes(self):
        df = make_valid_df()
        validate_raw_data(df, SAMPLE_CONFIG)  # should not raise

    def test_too_few_rows_raises(self):
        df = make_valid_df(n=5)  # below min_rows=10
        with pytest.raises(ValueError, match="Row count"):
            validate_raw_data(df, SAMPLE_CONFIG)

    def test_missing_target_column_raises(self):
        df = make_valid_df().drop(columns=["Late_delivery_risk"])
        with pytest.raises(ValueError, match="Target column"):
            validate_raw_data(df, SAMPLE_CONFIG)

    def test_non_binary_target_raises(self):
        df = make_valid_df()
        df["Late_delivery_risk"] = df["Late_delivery_risk"].astype(float)
        df.loc[0, "Late_delivery_risk"] = 2.0  # invalid value
        with pytest.raises(ValueError, match="non-binary"):
            validate_raw_data(df, SAMPLE_CONFIG)

    def test_leakage_column_present_raises(self):
        df = make_valid_df()
        df["Delivery Status"] = "Late delivery"  # leakage column still present
        with pytest.raises(ValueError, match="Leakage"):
            validate_raw_data(df, SAMPLE_CONFIG)

    def test_high_null_rate_raises(self):
        df = make_valid_df()
        # Set 80% of a numeric feature to null — far above 10% threshold
        df.loc[:79, "Days for shipment (scheduled)"] = np.nan
        with pytest.raises(ValueError, match="null rate"):
            validate_raw_data(df, SAMPLE_CONFIG)

    def test_target_rate_too_low_raises(self):
        df = make_valid_df(n=100)
        # Force target rate to 10% — below min of 40%
        df["Late_delivery_risk"] = 0
        df.loc[:9, "Late_delivery_risk"] = 1
        with pytest.raises(ValueError, match="positive rate"):
            validate_raw_data(df, SAMPLE_CONFIG)

    def test_target_rate_too_high_raises(self):
        df = make_valid_df(n=100)
        # Force target rate to 95% — above max of 70%
        df["Late_delivery_risk"] = 1
        df.loc[:4, "Late_delivery_risk"] = 0
        with pytest.raises(ValueError, match="positive rate"):
            validate_raw_data(df, SAMPLE_CONFIG)


class TestValidateFeatureMatrix:
    def test_valid_matrix_passes(self):
        X = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]})
        validate_feature_matrix(X, ["feat_a", "feat_b"])  # should not raise

    def test_missing_column_raises(self):
        X = pd.DataFrame({"feat_a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="missing expected columns"):
            validate_feature_matrix(X, ["feat_a", "feat_b"])

    def test_null_values_raise(self):
        X = pd.DataFrame({"feat_a": [1.0, None], "feat_b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="Null values"):
            validate_feature_matrix(X, ["feat_a", "feat_b"])
