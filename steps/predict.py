"""
Inference steps.

Three steps compose the inference pipeline:
  load_inference_data → load_model → generate_predictions

Critical: generate_predictions uses the SAME core/ functions as the training
pipeline. No separate preprocessing code. Same sklearn Pipeline artifact.
This is how we prevent training-serving skew.
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Annotated
from sklearn.pipeline import Pipeline
from zenml import step
from zenml.client import Client

CONFIG_PATH = "configs/training_config.yaml"


@step(enable_cache=False)
def load_inference_data() -> pd.DataFrame:
    """
    Load raw CSV data to score.

    Uses the same source path as training by default.
    In production this would point to new, unscored orders.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    source_path = config["data"]["source_path"]
    encoding = config["data"]["encoding"]

    print(f"Loading inference data from: {source_path}")
    df = pd.read_csv(source_path, encoding=encoding)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    return df


@step
def load_model() -> Pipeline:
    """
    Fetch the latest fitted_pipeline artifact from the ZenML artifact store.

    This loads the exact sklearn Pipeline that was fitted on training data —
    same preprocessor statistics (scaler mean/std, encoder vocabulary),
    same model weights. No refitting. No new code paths.
    """
    client = Client()

    print("Fetching latest trained_pipeline artifact from ZenML store...")
    artifact_version = client.get_artifact_version("trained_pipeline")
    pipeline = artifact_version.load()

    model_type = type(pipeline.named_steps["model"]).__name__
    print(f"Loaded pipeline: preprocessor + {model_type}")
    print(f"Artifact version: {artifact_version.version} "
          f"(created: {artifact_version.created.strftime('%Y-%m-%d %H:%M')})")

    return pipeline


@step(enable_cache=False)
def generate_predictions(
    df: pd.DataFrame,
    pipeline: Pipeline,
) -> Annotated[pd.DataFrame, "predictions"]:
    """
    Apply feature engineering and model to produce predictions.

    Uses core.preprocessing functions — identical code path to training.
    Outputs a DataFrame with risk scores and binary flags, saved to CSV.

    Returns:
        DataFrame with columns: order_id, late_delivery_probability,
        late_delivery_predicted, plus key context columns.
    """
    from core.preprocessing import drop_unwanted_columns, extract_date_features

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    target_col = config["data"]["target_column"]
    date_col = config["data"]["date_column"]
    threshold = config["model"]["prediction_threshold"]

    # Keep original columns for output context before any dropping
    context_cols = ["Order Id", "Order Item Id", "Customer Id",
                    "order date (DateOrders)", "Shipping Mode", "Market",
                    "Order Region", "Customer Segment"]
    available_context = [c for c in context_cols if c in df.columns]
    df_context = df[available_context].copy().reset_index(drop=True)

    # Apply identical feature engineering as training pipeline
    df_processed = drop_unwanted_columns(df, config)
    df_processed = extract_date_features(df_processed, date_col)

    # Drop target if present (scoring new data may or may not have labels)
    has_labels = target_col in df_processed.columns
    if has_labels:
        y_true = df_processed[target_col].reset_index(drop=True)
        X = df_processed.drop(columns=[target_col])
    else:
        y_true = None
        X = df_processed

    # Keep only configured feature columns
    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]
    feature_cols = [c for c in numeric_cols + categorical_cols if c in X.columns]
    X = X[feature_cols].reset_index(drop=True)

    print(f"Scoring {len(X):,} orders with {X.shape[1]} features...")

    # Transform using frozen training statistics — never refit
    X_transformed = pd.DataFrame(
        pipeline.named_steps["preprocessor"].transform(X)
    )

    # Score
    model = pipeline.named_steps["model"]
    y_prob = model.predict_proba(X_transformed)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    flagged = y_pred.sum()
    print(f"Flagged {flagged:,} orders as at-risk ({flagged/len(y_pred):.1%} flag rate)")
    print(f"Mean predicted probability: {y_prob.mean():.3f}")

    # Build output
    output = df_context.copy()
    output["late_delivery_probability"] = y_prob.round(4)
    output["late_delivery_predicted"] = y_pred
    output["scored_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    if has_labels:
        output["late_delivery_actual"] = y_true.values
        from core.evaluation import compute_metrics
        metrics = compute_metrics(y_true, y_pred, y_prob, threshold)
        print(f"\nLabeled data — actual performance:")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")

    # Save to outputs/
    output_dir = Path(config["output"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"predictions_{timestamp}.csv"
    output.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return output
