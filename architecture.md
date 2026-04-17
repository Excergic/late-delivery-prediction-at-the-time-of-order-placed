# Architecture: Supply Chain Late Delivery Prediction

## MLOps Pipeline Overview

A production ML system is not a model — it is a system of pipelines. This project
decomposes into four distinct pipelines, each with a clear responsibility:

```
Training Pipeline:   raw data → validate → features → train → evaluate → register
Inference Pipeline:  new orders → load model → transform → predict → output CSV
Drift Pipeline:      current batch features vs training baseline → Evidently report
Monitoring Pipeline: reconcile predictions vs actual outcomes (weekly, with delay)
```

Critical rule enforced throughout: the inference pipeline uses the exact same
sklearn Pipeline object (preprocessor + model) as training. One artifact.
No recomputed statistics at serving time. No separate code paths.

---

## Data Plan

**Source:** `/Volumes/SeagateSSD/Projects/mlops-stack/data/supply_chain.csv`
- 180,519 rows × 53 columns
- Encoding: latin-1 (non-UTF-8 characters present)
- Labels: fully labeled (`Late_delivery_risk`: 1=late, 0=on time)
- Class balance: 54.8% late, 45.2% on time

**Versioning:** ZenML artifact store snapshots every pipeline run automatically.
Each model version is traceable to the exact data snapshot that produced it.

**Train/Test Split:** Time-based (not random).
- Train: orders before cutoff date (~80%)
- Test: orders after cutoff date (~20%)
- Reason: random split leaks future information. We always predict on new orders,
  never on past ones.

**Columns dropped at ingestion:**

| Category | Columns |
|---|---|
| Leakage | `Delivery Status`, `Days for shipping (real)`, `shipping date (DateOrders)` |
| PII | `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street` |
| Empty/useless | `Product Description` (100% null), `Order Zipcode` (86% null), `Product Image` |
| IDs | `Order Id`, `Order Item Id`, `Order Customer Id`, `Customer Id`, `Product Card Id`, `Order Item Cardprod Id` |
| Redundant | `Product Category Id` (duplicate of `Category Id`) |
| Post-order | `Order Status` (always PENDING at prediction time) |

**Data validation gates (fail = pipeline stops):**

| Check | Threshold |
|---|---|
| Required columns present | Hard stop if any missing |
| `Late_delivery_risk` values | Must be binary 0/1 only |
| Null rate on key features | < 10% per feature |
| Leakage columns absent | `Delivery Status` and `Days for shipping (real)` must be dropped |
| Row count | Between 1,000 and 500,000 |
| Target distribution | 40%–70% positive rate |

---

## Feature Engineering Plan

All preprocessing is bundled inside a single `sklearn.Pipeline` object.
The pipeline is fitted once on training data, serialized as an artifact,
and loaded identically at inference time. No recomputation at serving.

**Numeric features → StandardScaler:**
```
Days for shipment (scheduled), Order Item Discount, Order Item Discount Rate,
Order Item Product Price, Order Item Profit Ratio, Order Item Quantity,
Sales, Order Item Total, Order Profit Per Order, Benefit per order,
Sales per customer, Product Price, Latitude, Longitude, Product Status
```

**Low-cardinality categoricals → OneHotEncoder(handle_unknown='ignore'):**
```
Shipping Mode (4 values), Type (3 values), Customer Segment (3 values),
Market (5 values)
```

**Medium-cardinality categoricals → OneHotEncoder(handle_unknown='ignore'):**
```
Department Name (~10), Category Name (~45), Order Region (~23)
```

**High-cardinality → Dropped (first version):**
```
Customer City, Order City, Customer State, Order State,
Customer Country, Order Country, Product Name
```
Reason: hundreds–thousands of unique values, high noise risk, potential for
overfitting. Add frequency encoding in v2 if model performance demands it.

**Date features extracted from `order date (DateOrders)`:**
```
order_month (1–12), order_day_of_week (0–6),
order_is_weekend (0/1), order_quarter (1–4)
```
These feed into the numeric pipeline after extraction.

**Pipeline structure:**
```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(),                       numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',        <selected_model>),
])
```

**No feature store.** Single CSV source, batch-only, no multi-model feature reuse.

---

## Training & Evaluation Plan

**Baselines (tracked first in MLflow):**

| Baseline | Purpose |
|---|---|
| Predict all late (always 1) | Absolute floor: Recall=1.0, Precision=0.548 |
| Logistic Regression | ML floor: simplest model, fast signal on feature quality |

**Candidate models (tried in order if baseline insufficient):**
1. LightGBM — best default for tabular data, built-in feature importance
2. Random Forest — robust fallback
3. XGBoost — if LightGBM underperforms

**Evaluation metrics:**

| Metric | Role | Target |
|---|---|---|
| Recall | Primary | ≥ 0.80 |
| Precision | Guardrail | ≥ 0.65 |
| PR-AUC | Overall discrimination | Maximize |
| F1 | Balance measure | Track |
| Confusion matrix | Full error picture | Log as artifact |

**Threshold tuning:** LightGBM outputs probabilities. Default 0.5 threshold will
be tuned on the validation set to maximize recall while respecting the precision
guardrail. Final threshold is a business decision reviewed with the ops team.

**Slice evaluation (minimum pass criteria):**
- Recall ≥ 0.70 per `Shipping Mode` value
- Recall ≥ 0.70 per `Market` value
- Recall ≥ 0.70 per `order_month` (no seasonal blind spots)

**Hyperparameter tuning:** manual first (3–5 configs). Add Optuna only if:
- Manual tuning stalls within 2–3 recall points of target, or
- Feature engineering is stable and tuning is the bottleneck.

**Experiment tracking:** MLflow logs every run automatically:
- Parameters: model type, hyperparameters, threshold, random seed, feature list, train cutoff date
- Metrics: all above metrics + per-slice breakdowns
- Artifacts: fitted sklearn Pipeline, confusion matrix, PR curve, feature importance chart
- Tags: data snapshot date

---

## Deployment Plan

**Serving mode:** Batch — predictions generated on demand or on a schedule.
No real-time API needed. Ops team needs a flagged list each morning.

**Inference pipeline output:**
```
flagged_shipments_YYYY-MM-DD.csv

Columns:
  Order Id | Late_delivery_risk_pred | Late_probability | Risk_flag
  77202    | 1                       | 0.87             | HIGH
  75938    | 0                       | 0.21             | OK
```

**Rollout strategy (greenfield — no existing model to shadow):**

| Phase | Action |
|---|---|
| Week 1–2 | Run predictions alongside normal ops. Team reviews flags, makes independent decisions. Compare model flags vs actual outcomes. |
| Week 3+ | Ops team acts on model flags as primary signal. |
| Future updates | Canary: new model runs 1 week alongside current, compare metrics before promoting. |

**Model promotion flow:**
```
Training run → gates pass → register as "Staging"
                                  ↓
                     1 week real-order validation
                                  ↓
                     Manual approval → promote to "Production"
                                  ↓
                     Inference pipeline loads "Production" model
```

**Rollback:** promote previous archived version to Production. < 30 seconds.
Previous model is never deleted.

**Output destination:** CSV file (Phase 1). Migrate to database table when ops
team tooling requires it — only `df.to_csv()` → `df.to_sql()` changes needed.

---

## Monitoring & Drift Plan

**Tool:** Evidently (via ZenML integration). Each batch run produces drift
reports saved as ZenML artifacts.

**Layer 1 — Data health (every batch, no labels needed):**

| Monitor | Alert threshold |
|---|---|
| Null rate per feature | > 10% on any key feature |
| Row count | < 10 or > 100,000 rows |
| `Shipping Mode` distribution | PSI > 0.20 |
| `Market` distribution | PSI > 0.20 |
| `Days for shipment (scheduled)` | PSI > 0.25 (KS test) |
| Unknown categories | > 5% unknown in any categorical |

**Layer 2 — Prediction health (every batch, no labels needed):**

| Monitor | Alert threshold |
|---|---|
| Flag rate | Outside 40%–70% (training baseline: 54.8%) |
| Mean late probability | Drifts > 15% from training baseline |
| Flag rate by Shipping Mode | Any mode changes > 25% |

**Layer 3 — Model performance (weekly, labels required ~2–3 day delay):**

| Monitor | Alert threshold |
|---|---|
| Recall on last week's shipments | < 0.75 |
| Precision | < 0.55 |
| Confusion matrix | Logged weekly to MLflow |

**Retraining triggers:**

| Trigger | Type | Action |
|---|---|---|
| PSI > 0.25 on Shipping Mode or Market | Data drift | Investigate first, retrain if confirmed |
| Flag rate outside 40–70% for 3 consecutive days | Prediction drift | Check pipeline, then retrain |
| Weekly recall < 0.75 | Performance drop | Retrain with recent data |
| Scheduled: every 3 months | Calendar | Keep model fresh |

**Alert delivery:** ZenML Slack alerter (when Slack workspace available).
Initial fallback: warning logged to `alerts/YYYY-MM-DD.log` alongside output CSV.

---

## Versioning & Governance

**Model registry:** MLflow (via ZenML)

**Lifecycle stages:**
```
Candidate → Staging → Production → Archived
```

**Promotion gates:**

Candidate → Staging:
- Recall ≥ 0.80 on time-based test set
- Precision ≥ 0.65
- No slice recall below 0.70 (Shipping Mode, Market)
- Data validation clean (no schema violations)

Staging → Production:
- 1 week real-order prediction review
- Flag rate within 40–70%
- Manual approval (CLI command)

**Audit trail per version:**
- Training data date range
- Git commit hash
- All metrics (global + per-slice)
- Fitted sklearn Pipeline artifact (checksummed)
- Feature list
- Evidently drift baseline report
- Promoter identity + timestamp

**Retention:** production versions kept indefinitely. Experimental versions
archived after 90 days but never deleted.

---

## ZenML Stack Specification

| Component | Choice | Rationale |
|---|---|---|
| **Orchestrator** | Local | Single machine, no cloud infrastructure needed yet |
| **Artifact Store** | Local | All artifacts on local filesystem, auto-versioned |
| **Experiment Tracker** | MLflow (local) | Full run comparison, metric charts, self-hosted, free |
| **Data Validator** | Evidently | Drift reports + quality checks, native ZenML integration |
| **Model Registry** | MLflow | Stage transitions, lineage, rollback |
| **Alerter** | None → Slack (later) | No workspace configured yet |
| **ZenML Server** | Local OSS (`zenml login --local`) | Dashboard UI for pipeline visualization |

**Not included:**
- Container Registry — local orchestrator doesn't containerize steps
- Model Deployer — batch inference loads from registry directly, no API
- Feature Store — single CSV source, no multi-model reuse
- Step Operator — no GPU workloads
- Cloud components — add when team or data volume grows

---

## Pipeline Decomposition

**Training pipeline** (`pipelines/training.py`):
```
load_data → validate_data → preprocess_features → train_model →
evaluate_model → detect_drift_baseline → register_model
```

**Inference pipeline** (`pipelines/inference.py`):
```
load_new_orders → validate_input → load_production_model →
generate_predictions → write_output_csv
```

**Drift detection pipeline** (`pipelines/drift.py`):
```
load_current_batch → load_reference_profile → run_evidently →
evaluate_thresholds → log_alerts
```

**Monitoring pipeline** (`pipelines/monitoring.py`):
```
load_predictions_window → load_actuals → compute_metrics →
compare_vs_baseline → log_weekly_report
```

---

## Project Structure

```
supply-chain-ml-system/
├── configs/
│   └── training_config.yaml       # hyperparams, feature lists, thresholds
├── core/                          # pure Python logic — NO framework imports
│   ├── __init__.py
│   ├── preprocessing.py           # feature engineering, pipeline building
│   ├── validation.py              # data quality checks
│   └── evaluation.py              # metric computation, slice evaluation
├── steps/                         # ZenML steps — thin wrappers over core/
│   ├── ingest.py
│   ├── validate.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── drift.py
│   └── register.py
├── pipelines/
│   ├── training.py
│   ├── inference.py
│   ├── drift.py
│   └── monitoring.py
├── tests/
│   ├── test_preprocessing.py      # imports from core/ only — no ZenML needed
│   ├── test_validation.py
│   └── test_evaluation.py
├── data/
│   └── supply_chain.csv           # symlink or copy of source data
├── outputs/                       # batch prediction CSVs land here
├── alerts/                        # drift/monitoring alert logs
├── pyproject.toml
├── problem_statement.md
└── architecture.md
```

**Key design principle:** `core/` contains pure Python logic with no framework
imports. Steps import from `core/`. Tests import from `core/`. This means tests
run without ZenML/MLflow installed, and migrating to a different framework only
requires rewriting `steps/` — not `core/`.

---

## MVP Scope

**Build first (Phases 3.1–3.6):**
1. Project setup + ZenML stack init
2. Data loading + validation step
3. EDA + feature understanding
4. Preprocessing pipeline (sklearn Pipeline)
5. Training pipeline with MLflow tracking
6. Evaluation with slice metrics + model registration

**Add before production (Phases 3.7–3.10):**
7. Drift detection pipeline (Evidently)
8. Inference pipeline (batch CSV output)
9. Monitoring pipeline (weekly reconciliation)
10. Tests + production hardening

**Deferred (post-MVP):**
- Real-time API endpoint
- Cloud artifact store (S3/GCS)
- Slack alerter
- Frequency encoding for high-cardinality features
- Automated retraining trigger (run manually first, automate after 3 months)

---

## Success Criteria

The system is production-ready when:
1. Recall ≥ 0.80 on held-out time-based test set
2. Precision ≥ 0.65
3. No slice recall below 0.70 (Shipping Mode, Market)
4. Training pipeline reproducible end-to-end via single ZenML command
5. Inference pipeline outputs a clean CSV the ops team can act on
6. Drift detection operational and producing weekly reports
7. Model version traceable to exact data, code, and metrics that produced it
8. Rollback tested and confirmed working
