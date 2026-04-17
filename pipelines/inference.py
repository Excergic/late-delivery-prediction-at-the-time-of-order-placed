"""
Inference pipeline.

Loads new orders → fetches trained model artifact → scores → saves CSV.

The fitted_pipeline artifact carries both the preprocessor (frozen statistics
from training) and the model. No refitting happens here — ever.

Run with:
    uv run python run_inference.py
"""

from zenml import pipeline

from steps.predict import load_inference_data, load_model, generate_predictions


@pipeline(name="supply_chain_inference")
def inference_pipeline() -> None:
    """
    Batch inference pipeline.

    Steps:
    1. load_inference_data  — read CSV of orders to score
    2. load_model           — fetch fitted_pipeline from ZenML artifact store
    3. generate_predictions — feature engineering + scoring + CSV output
    """
    df = load_inference_data()
    trained_pipeline = load_model()
    generate_predictions(df=df, pipeline=trained_pipeline)
