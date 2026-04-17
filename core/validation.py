"""
Data validation logic.

All checks are pure Python — no ZenML, no MLflow.
Raises ValueError with a clear message on any failure.
The ZenML step wraps these functions and handles logging.
"""

from __future__ import annotations

import pandas as pd


def validate_raw_data(df: pd.DataFrame, config: dict) -> None:
    """
    Validate raw data before any processing.
    Raises ValueError immediately on any failure — fail fast.

    Checks:
    - Row count within bounds
    - Required target column present and binary
    - Leakage columns absent (must have been dropped before calling this)
    - Null rate on key features below threshold
    - Target distribution within expected range
    """
    target = config["data"]["target_column"]
    min_rows = config["data"]["min_rows"]
    max_rows = config["data"]["max_rows"]
    target_min = config["data"]["target_min_rate"]
    target_max = config["data"]["target_max_rate"]

    # Row count
    n = len(df)
    if not (min_rows <= n <= max_rows):
        raise ValueError(
            f"Row count {n:,} outside expected range [{min_rows:,}, {max_rows:,}]. "
            "Check for partial data pull or wrong file."
        )

    # Target column present
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' missing from data. "
            "Check column names for renames or schema changes."
        )

    # Target is binary 0/1
    unique_vals = set(df[target].dropna().unique())
    if not unique_vals.issubset({0, 1}):
        raise ValueError(
            f"Target column '{target}' contains non-binary values: {unique_vals}. "
            "Expected only 0 and 1."
        )

    # Leakage columns must be absent
    leakage_cols = config["features"]["leakage_columns"]
    present_leakage = [c for c in leakage_cols if c in df.columns]
    if present_leakage:
        raise ValueError(
            f"Leakage columns still present in data: {present_leakage}. "
            "These must be dropped before validation."
        )

    # Null rate on numeric and categorical features
    feature_cols = config["features"]["numeric"] + config["features"]["categorical"]
    present_features = [c for c in feature_cols if c in df.columns]
    for col in present_features:
        null_rate = df[col].isnull().mean()
        if null_rate > 0.10:
            raise ValueError(
                f"Feature '{col}' has {null_rate:.1%} null rate (threshold: 10%). "
                "Upstream data pipeline may have failed."
            )

    # Target distribution
    positive_rate = df[target].mean()
    if not (target_min <= positive_rate <= target_max):
        raise ValueError(
            f"Target positive rate {positive_rate:.1%} outside expected range "
            f"[{target_min:.0%}, {target_max:.0%}]. "
            "Check for wrong dataset or label corruption."
        )


def validate_feature_matrix(X: pd.DataFrame, expected_columns: list[str]) -> None:
    """
    Validate the feature matrix after preprocessing.
    Ensures all expected columns are present before model sees data.
    """
    missing = [c for c in expected_columns if c not in X.columns]
    if missing:
        raise ValueError(
            f"Feature matrix missing expected columns after preprocessing: {missing}. "
            "Feature engineering step may have changed."
        )

    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            f"Null values remain in feature matrix after preprocessing: {null_cols}. "
            "Imputation step did not cover all columns."
        )
