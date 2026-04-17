"""
Feature engineering and preprocessing logic.

All logic lives here — no ZenML, no MLflow.
The ZenML step calls build_pipeline() and fit_pipeline().

Key design: one sklearn Pipeline object for both training and inference.
The same object is serialized after training and loaded at inference time.
No statistics are recomputed at serving time. No skew.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# --------------------------------------------------------------------------- #
# Date feature extraction                                                      #
# --------------------------------------------------------------------------- #

def extract_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Extract order date features from a datetime column.
    Returns df with new columns added and original date column dropped.

    New columns:
        order_month        — 1–12 (seasonality: holiday rush, peak periods)
        order_day_of_week  — 0–6  (Friday orders often ship Monday)
        order_is_weekend   — 0/1  (weekend orders frequently delayed)
        order_quarter      — 1–4  (Q4 holiday volume spike)
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["order_month"] = dt.dt.month.astype("float32")
    df["order_day_of_week"] = dt.dt.dayofweek.astype("float32")
    df["order_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float32")
    df["order_quarter"] = dt.dt.quarter.astype("float32")
    df = df.drop(columns=[date_col])
    return df


# --------------------------------------------------------------------------- #
# Column cleanup                                                               #
# --------------------------------------------------------------------------- #

def drop_unwanted_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop leakage, PII, useless, and ID columns defined in config.
    Only drops columns that are actually present — no error on missing.
    """
    to_drop = (
        config["features"]["leakage_columns"]
        + config["features"]["pii_columns"]
        + config["features"]["useless_columns"]
        + config["features"]["id_columns"]
    )
    present = [c for c in to_drop if c in df.columns]
    return df.drop(columns=present)


# --------------------------------------------------------------------------- #
# Pipeline construction                                                        #
# --------------------------------------------------------------------------- #

def build_pipeline(model, config: dict) -> Pipeline:
    """
    Build the full sklearn Pipeline: preprocessor + model.

    This pipeline is the single source of truth for feature transformation.
    It is fit once on training data, serialized as an artifact, and loaded
    identically at inference time. The StandardScaler's mean/std and the
    OneHotEncoder's category vocabulary are frozen inside the fitted pipeline.

    Args:
        model: an unfitted sklearn-compatible estimator
        config: the training config dict (from training_config.yaml)

    Returns:
        Unfitted sklearn Pipeline ready for .fit()
    """
    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",  # unseen categories at serving → all zeros
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",  # drop any columns not in numeric or categorical lists
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def prepare_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply all feature engineering to raw (but cleaned) data.
    Returns X (features) and y (target).

    Steps:
    1. Drop unwanted columns
    2. Extract date features
    3. Separate target from features
    """
    date_col = config["data"]["date_column"]
    target_col = config["data"]["target_column"]

    df = drop_unwanted_columns(df, config)
    df = extract_date_features(df, date_col)

    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y
