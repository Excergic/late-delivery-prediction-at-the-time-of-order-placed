"""
Model training step.

Fits the model on preprocessed training data and logs everything to MLflow:
parameters, training metrics, and the fitted pipeline artifact.

The pipeline already has a fitted preprocessor from the preprocess step.
This step fits the model component on the transformed training data.
"""

import yaml
import pandas as pd
from typing import Annotated
from sklearn.pipeline import Pipeline
from zenml import step

CONFIG_PATH = "configs/training_config.yaml"


@step(experiment_tracker="mlflow_tracker")
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
) -> Annotated[Pipeline, "trained_pipeline"]:
    """
    Fit the model on preprocessed training data.

    Logs to MLflow:
    - All hyperparameters from config
    - Training set size and positive rate
    - Model type

    Returns:
        Fully fitted pipeline (preprocessor + model), ready for evaluation
    """
    import mlflow

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    model_type = config["model"]["type"]
    seed = config["model"]["random_seed"]

    print(f"Training {model_type} on {len(X_train):,} samples...")
    print(f"Positive rate: {y_train.mean():.1%}")

    # Log parameters to MLflow
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("random_seed", seed)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("train_positive_rate", round(float(y_train.mean()), 4))
    mlflow.log_param("train_cutoff_date", config["data"]["train_cutoff_date"])
    mlflow.log_param("n_numeric_features", len(config["features"]["numeric"]))
    mlflow.log_param("n_categorical_features", len(config["features"]["categorical"]))
    mlflow.log_param("prediction_threshold", config["model"]["prediction_threshold"])

    # Log model-specific hyperparameters
    model_cfg = config["model"].get(model_type, {})
    for k, v in model_cfg.items():
        mlflow.log_param(f"{model_type}__{k}", v)

    # Fit just the model component on already-transformed X_train
    # The preprocessor was fitted in the preprocess step — we don't refit it
    model = pipeline.named_steps["model"]
    model.fit(X_train, y_train)
    print(f"{model_type} training complete ✓")

    return pipeline
