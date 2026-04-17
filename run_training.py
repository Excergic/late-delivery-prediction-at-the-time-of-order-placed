"""
Entry point for the training pipeline.

Run with:
    uv run python run_training.py
"""

import yaml
from pipelines.training import training_pipeline


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    print("Starting supply-chain-stack training pipeline...")
    print(f"Model: {config['model']['type']}")
    print(f"Train cutoff: {config['data']['train_cutoff_date']}")
    print()

    training_pipeline()


if __name__ == "__main__":
    main()
