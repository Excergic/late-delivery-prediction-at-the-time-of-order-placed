"""
Entry point for the inference pipeline.

Run with:
    uv run python run_inference.py
"""

from pipelines.inference import inference_pipeline


def main():
    print("Starting supply-chain-stack inference pipeline...")
    print()
    inference_pipeline()


if __name__ == "__main__":
    main()
