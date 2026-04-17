"""
Entry point for the drift detection pipeline.

Run with:
    uv run python run_drift_detection.py
"""

from pipelines.drift import drift_detection_pipeline


def main():
    print("Starting supply-chain-stack drift detection pipeline...")
    print()
    drift_detection_pipeline()


if __name__ == "__main__":
    main()
