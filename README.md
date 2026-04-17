# Supply Chain Late Delivery Prediction

Predicts which shipments are at risk of arriving late so the ops team can flag them for extra care before dispatch. Binary classification on 180K orders across global markets.

**Primary metric:** Recall (missing a late shipment costs more than a false alarm)
**Model:** LightGBM · Recall 0.732 · Precision 0.653 · PR-AUC 0.807
**Stack:** ZenML · MLflow · Evidently · scikit-learn

---

## Pipelines

| Pipeline | Entry point | What it does |
|---|---|---|
| Training | `uv run python run_training.py` | Ingest → split → fit preprocessor → train → evaluate → gate check |
| Inference | `uv run python run_inference.py` | Load model artifact → score all orders → write `outputs/predictions_*.csv` |
| Drift detection | `uv run python run_drift_detection.py` | Compare current order distributions vs training baseline → Evidently HTML report |

---

## Project Structure

```
supply-chain-ml-system/
├── configs/
│   └── training_config.yaml     # All thresholds, feature lists, hyperparameters
├── core/                        # Pure Python logic — no framework imports
│   ├── preprocessing.py         # Feature engineering, pipeline builder
│   ├── validation.py            # Data quality checks
│   └── evaluation.py            # Metrics, slice metrics, promotion gates
├── steps/                       # ZenML steps (thin wrappers around core/)
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py               # Inference steps
│   └── drift.py                 # Drift detection steps
├── pipelines/
│   ├── training.py
│   ├── inference.py
│   └── drift.py
├── tests/                       # Unit tests — no ZenML, no CSV needed
│   ├── test_validation.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   ├── test_inference.py
│   └── test_drift.py
├── data/
│   └── supply_chain.csv
├── outputs/
│   ├── predictions_*.csv        # Scored orders
│   └── drift_reports/           # Evidently HTML reports
├── run_training.py
├── run_inference.py
├── run_drift_detection.py
└── pyproject.toml
```

---

## Setup

```bash
# Install dependencies (Python 3.11 required)
uv sync --python 3.11

# Initialise ZenML stack (first time only)
zenml init
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-registry register mlflow_registry --flavor=mlflow
zenml data-validator register evidently_validator --flavor=evidently
zenml stack register supply-chain-stack \
  -o default \
  -a default \
  -e mlflow_tracker \
  -r mlflow_registry \
  -dv evidently_validator
zenml stack set supply-chain-stack
zenml integration install mlflow evidently -y --uv
```

---

## Running the Tests

```bash
uv run pytest tests/ -v
# 59 tests, ~2s, no external dependencies
```

---

## Configuration

All tuneable values live in `configs/training_config.yaml`. Key settings:

| Key | Value | Notes |
|---|---|---|
| `data.train_cutoff_date` | `2017-01-01` | Orders before = train (125K), after = test (55K) |
| `model.type` | `lightgbm` | Also supports `logistic_regression`, `random_forest` |
| `model.prediction_threshold` | `0.35` | Optimised for recall ≥ 0.65 precision floor |
| `evaluation.min_recall` | `0.70` | Promotion gate — pipeline stops if not met |
| `evaluation.min_precision` | `0.65` | Promotion gate |
| `evaluation.min_slice_recall` | `0.35` | Per Shipping Mode / Market slice floor |
| `monitoring.drift_share_warning` | `0.30` | Warn if ≥30% of features drift |
| `monitoring.drift_share_critical` | `0.50` | Halt if ≥50% of features drift |

---

## Key Design Decisions

**Time-based split, not random.** Orders before 2017-01-01 are training data; after is test. Random splits leak future information into training when records are correlated over time.

**Preprocessor fitted on training data only.** The `sklearn.Pipeline` artifact carries frozen scaler statistics and encoder vocabulary from the training period. The inference pipeline loads this artifact and transforms new data with the same statistics — never refitting. This prevents training-serving skew.

**`core/` module isolation.** All business logic (feature engineering, metrics, gates) lives in `core/` with no framework imports. Steps are thin wrappers. Tests run in 2s with no ZenML, no MLflow, no CSV.

**Leakage columns dropped before any processing.** `Delivery Status` and `Days for shipping (real)` tell you the outcome — they would give the model perfect predictions in training and then be unavailable at inference time when decisions actually need to be made.

**Standard Class recall is low by design.** Standard Class shipments have a 38% late rate vs 96% for First Class. At threshold 0.35, the model assigns probability ≈ 0.38 to most Standard Class orders — right at the boundary. The slice recall floor (0.35) reflects this data reality. Raising it to 0.70 would require dropping precision below the business floor.

**Promotion gates block bad models.** The training pipeline raises before the model can be registered if recall, precision, or any slice recall gate fails. A subpar model never reaches inference.
