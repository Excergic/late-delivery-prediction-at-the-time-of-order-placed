"""
Data ingestion step.

Reads the supply chain CSV, drops unwanted columns, then validates.
Config is loaded directly from YAML — not passed as a step parameter,
which avoids ZenML serialization issues with complex nested dicts.
"""

import yaml
import pandas as pd
from zenml import step

from core.preprocessing import drop_unwanted_columns
from core.validation import validate_raw_data

CONFIG_PATH = "configs/training_config.yaml"


@step
def ingest_data() -> pd.DataFrame:
    """
    Load and validate raw supply chain data.

    Returns cleaned DataFrame as a ZenML artifact.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    source_path = config["data"]["source_path"]
    encoding = config["data"]["encoding"]

    print(f"Loading data from: {source_path}")
    df = pd.read_csv(source_path, encoding=encoding)
    print(f"Raw data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    df = drop_unwanted_columns(df, config)
    print(f"After dropping unwanted columns: {df.shape[0]:,} rows × {df.shape[1]} columns")

    validate_raw_data(df, config)
    print("Data validation passed ✓")

    target = config["data"]["target_column"]
    positive_rate = df[target].mean()
    print(f"Target distribution: {positive_rate:.1%} late ({df[target].sum():,} / {len(df):,})")

    return df
