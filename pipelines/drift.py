"""
Drift detection pipeline.

Compares incoming order distributions against the training baseline.
Run on a schedule (e.g. daily) or after a new batch of orders arrives.

If critical drift is detected, the step raises — signalling that the model
should be investigated and potentially retrained before the next inference run.

Run with:
    uv run python run_drift_detection.py
"""

from zenml import pipeline

from steps.drift import build_reference_dataset, build_current_dataset, run_drift_detection


@pipeline(name="supply_chain_drift_detection")
def drift_detection_pipeline() -> None:
    """
    Drift detection pipeline.

    Steps:
    1. build_reference_dataset — training period feature distributions (the baseline)
    2. build_current_dataset   — current/recent orders feature distributions
    3. run_drift_detection     — Evidently K-S / chi-square tests, HTML report, alert
    """
    reference = build_reference_dataset()
    current = build_current_dataset()
    run_drift_detection(reference=reference, current=current)
