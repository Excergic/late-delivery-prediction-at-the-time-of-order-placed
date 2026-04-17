"""
Drift detection steps.

Compares the distribution of incoming orders against the training baseline.
Uses Evidently DataDriftPreset: K-S test for numeric features, chi-square
for categorical. A low p-value (< 0.05) means the distribution has shifted.

Three steps:
  build_reference_dataset → build_current_dataset → run_drift_detection

Run on a schedule (daily/weekly) or trigger-based after new data arrives.
"""

import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Annotated
from zenml import step

CONFIG_PATH = "configs/training_config.yaml"
DRIFT_PVALUE_THRESHOLD = 0.05  # standard statistical threshold


def _prepare_feature_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply the same feature engineering as training. Returns raw feature matrix (no scaling)."""
    from core.preprocessing import drop_unwanted_columns, extract_date_features

    target_col = config["data"]["target_column"]
    date_col = config["data"]["date_column"]

    df = drop_unwanted_columns(df, config)
    df = extract_date_features(df, date_col)

    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]
    feature_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]

    return df[feature_cols].reset_index(drop=True)


@step(enable_cache=False)
def build_reference_dataset() -> Annotated[pd.DataFrame, "drift_reference"]:
    """
    Build the reference dataset from training data (before cutoff date).

    This is the distribution the model was trained on. All drift comparisons
    are relative to this baseline — if the incoming data shifts away from
    these distributions, the model's assumptions may no longer hold.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    source_path = config["data"]["source_path"]
    encoding = config["data"]["encoding"]
    date_col = config["data"]["date_column"]
    cutoff = pd.Timestamp(config["data"]["train_cutoff_date"])

    df = pd.read_csv(source_path, encoding=encoding)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df_train = df[dates < cutoff].copy()

    print(f"Reference (training) period: {dates[dates < cutoff].min().date()} → {cutoff.date()}")
    print(f"Reference rows: {len(df_train):,}")

    X_ref = _prepare_feature_dataset(df_train, config)
    print(f"Reference feature matrix: {X_ref.shape}")

    return X_ref


@step(enable_cache=False)
def build_current_dataset() -> Annotated[pd.DataFrame, "drift_current"]:
    """
    Build the current dataset from the most recent incoming orders (post-cutoff).

    In production this would read from a live data source or a recent batch.
    For now it reads the held-out test period, which represents orders the
    model will encounter in deployment.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    source_path = config["data"]["source_path"]
    encoding = config["data"]["encoding"]
    date_col = config["data"]["date_column"]
    cutoff = pd.Timestamp(config["data"]["train_cutoff_date"])

    df = pd.read_csv(source_path, encoding=encoding)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df_current = df[dates >= cutoff].copy()

    print(f"Current period: {cutoff.date()} → {dates[dates >= cutoff].max().date()}")
    print(f"Current rows: {len(df_current):,}")

    X_curr = _prepare_feature_dataset(df_current, config)
    print(f"Current feature matrix: {X_curr.shape}")

    return X_curr


@step(enable_cache=False)
def run_drift_detection(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> Annotated[dict, "drift_summary"]:
    """
    Run Evidently DataDriftPreset and alert on drift.

    For each feature:
    - Numeric: Kolmogorov-Smirnov test. p-value < 0.05 = drift.
    - Categorical: chi-square test. p-value < 0.05 = drift.

    Raises ValueError if critical drift threshold is exceeded.
    Saves an HTML report to outputs/drift_reports/.

    Returns:
        Summary dict with drifted feature list, counts, and per-feature p-values.
    """
    from evidently.future.report import Report
    from evidently.future.presets import DataDriftPreset

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    drift_share_warning = config["monitoring"].get("drift_share_warning", 0.3)
    drift_share_critical = config["monitoring"].get("drift_share_critical", 0.5)

    print(f"Running drift detection: {len(reference):,} reference rows vs {len(current):,} current rows")
    print(f"Features: {reference.shape[1]}")

    # Run Evidently report
    report = Report([DataDriftPreset(threshold=DRIFT_PVALUE_THRESHOLD)])
    snapshot = report.run(reference_data=reference, current_data=current)

    # Parse results
    results = json.loads(snapshot.json())
    metrics = results["metrics"]

    # Extract summary counts
    drifted_count = 0
    drifted_share = 0.0
    per_feature = {}

    for m in metrics:
        name = m["metric_name"]
        value = m["value"]

        if name.startswith("DriftedColumnsCount"):
            drifted_count = int(m["value"]["count"])
            drifted_share = float(m["value"]["share"])

        elif name.startswith("ValueDrift"):
            # Parse "ValueDrift(column=X,method=Y,threshold=T)"
            col_start = name.index("column=") + 7
            col_end = name.index(",", col_start)
            col_name = name[col_start:col_end]
            p_value = float(value)
            per_feature[col_name] = {
                "p_value": round(p_value, 6),
                "drifted": p_value < DRIFT_PVALUE_THRESHOLD,
            }

    total_features = reference.shape[1]
    drifted_features = [col for col, v in per_feature.items() if v["drifted"]]
    not_drifted = [col for col, v in per_feature.items() if not v["drifted"]]

    print(f"\nDrift detection results:")
    print(f"  Features analysed:  {total_features}")
    print(f"  Drifted features:   {drifted_count} ({drifted_share:.1%})")
    print(f"  Warning threshold:  {drift_share_warning:.0%} of features")
    print(f"  Critical threshold: {drift_share_critical:.0%} of features")

    if drifted_features:
        print(f"\n  Drifted features (p < {DRIFT_PVALUE_THRESHOLD}):")
        for col in sorted(drifted_features, key=lambda c: per_feature[c]["p_value"]):
            print(f"    {col}: p={per_feature[col]['p_value']:.4f}")

    if not_drifted:
        print(f"\n  Stable features: {', '.join(sorted(not_drifted))}")

    # Save HTML report
    report_dir = Path(config["output"].get("drift_reports_dir", "outputs/drift_reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"drift_report_{timestamp}.html"
    snapshot.save_html(str(report_path))
    print(f"\nDrift report saved to: {report_path}")

    # Build summary
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "total_features": total_features,
        "drifted_count": drifted_count,
        "drifted_share": round(drifted_share, 4),
        "drifted_features": drifted_features,
        "per_feature": per_feature,
        "report_path": str(report_path),
        "status": "ok",
    }

    # Alert logic
    if drifted_share >= drift_share_critical:
        summary["status"] = "critical"
        print(f"\n✗ CRITICAL DRIFT: {drifted_share:.1%} of features drifted "
              f"(threshold: {drift_share_critical:.0%})")
        print("  Model predictions may be unreliable. Investigate and consider retraining.")
        raise ValueError(
            f"Critical drift detected: {drifted_count}/{total_features} features drifted "
            f"({drifted_share:.1%} ≥ {drift_share_critical:.0%} threshold). "
            f"Drifted: {', '.join(drifted_features)}"
        )
    elif drifted_share >= drift_share_warning:
        summary["status"] = "warning"
        print(f"\n⚠ WARNING: {drifted_share:.1%} of features drifted "
              f"(warning at {drift_share_warning:.0%}). Monitor closely.")
    else:
        print(f"\n✓ No significant drift detected ({drifted_share:.1%} of features drifted)")

    return summary
