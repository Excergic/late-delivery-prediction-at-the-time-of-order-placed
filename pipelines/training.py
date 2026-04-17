"""
Training pipeline.

Orchestrates the full training lifecycle:
  ingest_data → preprocess_data → train_model → evaluate_model

Each step produces ZenML artifacts that are versioned and cached.
MLflow logs all parameters, metrics, and artifacts for every run.

Run with:
    uv run python run_training.py
"""

from zenml import pipeline

from steps.ingest import ingest_data
from steps.preprocess import preprocess_data
from steps.train import train_model
from steps.evaluate import evaluate_model


@pipeline(name="supply_chain_training")
def training_pipeline() -> None:
    """
    End-to-end training pipeline.

    Each step loads config from configs/training_config.yaml directly.
    ZenML multi-output steps are wired by positional order — the Annotated
    names are used as artifact labels but outputs are referenced by index
    when passed between steps.

    Steps:
    1. ingest_data      — load CSV, validate, return clean DataFrame
    2. preprocess_data  — time split, feature engineering, fit sklearn Pipeline
    3. train_model      — fit model, log params to MLflow
    4. evaluate_model   — compute metrics, check gates, log to MLflow
    """
    # Step 1 — Load and validate
    df = ingest_data()

    # Step 2 — Split, feature engineer, fit preprocessor
    X_train, X_test, y_train, y_test, fitted_pipeline, X_test_raw = preprocess_data(
        df=df
    )

    # Step 3 — Train model on transformed data
    trained_pipeline = train_model(
        X_train=X_train,
        y_train=y_train,
        pipeline=fitted_pipeline,
    )

    # Step 4 — Evaluate: global metrics, slice metrics, promotion gates
    evaluate_model(
        X_test=X_test,
        y_test=y_test,
        X_test_raw=X_test_raw,
        pipeline=trained_pipeline,
    )
