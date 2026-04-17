"""
Preprocessing step.

Time-based split → feature engineering → fit sklearn Pipeline on train only.
Returns transformed splits and the fitted pipeline as ZenML artifacts.
"""

import yaml
import pandas as pd
import numpy as np
from typing import Annotated
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from zenml import step

from core.preprocessing import prepare_features, build_pipeline

CONFIG_PATH = "configs/training_config.yaml"


def _get_model(config: dict):
    model_type = config["model"]["type"]
    seed = config["model"]["random_seed"]

    if model_type == "logistic_regression":
        cfg = config["model"]["logistic_regression"]
        return LogisticRegression(
            max_iter=cfg["max_iter"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
        )
    elif model_type == "lightgbm":
        cfg = config["model"]["lightgbm"]
        return lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            num_leaves=cfg["num_leaves"],
            min_child_samples=cfg["min_child_samples"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            class_weight=cfg["class_weight"],
            random_state=seed,
            verbosity=-1,
        )
    elif model_type == "random_forest":
        cfg = config["model"]["random_forest"]
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@step
def preprocess_data(df: pd.DataFrame) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[Pipeline, "fitted_pipeline"],
    Annotated[pd.DataFrame, "X_test_raw"],
]:
    """
    Time-based split, feature engineering, and preprocessor fitting.

    Critical: preprocessor is fit on training data ONLY.
    Scaler mean/std and encoder vocabulary are frozen at this step.
    X_test is transformed using training statistics — never refitted.

    Returns:
        X_train, X_test (transformed), y_train, y_test,
        fitted_pipeline (preprocessor + unfitted model),
        X_test_raw (pre-transform, for slice evaluation)
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    date_col = config["data"]["date_column"]
    cutoff = pd.Timestamp(config["data"]["train_cutoff_date"])

    dates = pd.to_datetime(df[date_col], errors="coerce")
    train_mask = dates < cutoff
    test_mask = dates >= cutoff

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"Train: {len(df_train):,} rows ({train_mask.mean():.1%})")
    print(f"Test:  {len(df_test):,} rows ({test_mask.mean():.1%})")

    X_train, y_train = prepare_features(df_train, config)
    X_test, y_test = prepare_features(df_test, config)

    print(f"Feature matrix — train: {X_train.shape}, test: {X_test.shape}")
    print(f"Target rate — train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")

    model = _get_model(config)
    pipeline = build_pipeline(model, config)

    # Fit preprocessor on training data only
    pipeline.named_steps["preprocessor"].fit(X_train)
    print("Preprocessor fitted on training data ✓")

    # Transform both splits using frozen training statistics
    X_train_t = pd.DataFrame(
        pipeline.named_steps["preprocessor"].transform(X_train)
    )
    X_test_t = pd.DataFrame(
        pipeline.named_steps["preprocessor"].transform(X_test)
    )

    print(f"Transformed — train: {X_train_t.shape}, test: {X_test_t.shape}")

    X_test_raw = X_test.reset_index(drop=True)

    return (
        X_train_t,
        X_test_t,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        pipeline,
        X_test_raw,
    )
