# Supply Chain Late Delivery Prediction

A machine learning system that predicts **which shipments will arrive late before they leave the warehouse** — so the operations team can act early instead of apologising to customers after the fact.

**Live demo →** [Frontend on Vercel](https://late-delivey-prediction.vercel.app)
**API →** [Backend on Render](https://late-delivery-prediction-at-the-time-of.onrender.com/docs)

---

## The Problem

### What was happening before

A global supply chain company ships tens of thousands of orders every month across markets in the US, Europe, Latin America, Asia, and Africa. Roughly **55% of their shipments arrive late**.

The operations team had no way to know in advance which orders would be delayed. They only found out when an angry customer called. By then it was too late — the shipment had already left, the customer was already disappointed, and all the team could do was apologise.

**This is a reactive system. Reactive is expensive.** Every late delivery means:
- A customer support call
- A possible refund or discount
- Damage to brand reputation
- A customer who might not return

### What this system does

When a new order is placed, this system looks at everything known about that order at that exact moment — the shipping method chosen, the product, the region, the discount, the customer type — and within milliseconds answers one question:

> **Is this shipment likely to arrive late?**

If the answer is yes, the order is flagged immediately. The ops team sees it, gives it priority handling, and the shipment has a real chance of going out on time.

**This turns a reactive problem into a proactive one.**

### Why this is hard

The tricky part is that the model cannot cheat. It cannot look at information that only exists after the delivery — like whether it actually arrived late, or how long it actually took. It can only use information available at the moment the customer clicks "Place Order."

Think of it like a doctor diagnosing a patient: they can only use symptoms visible today, not test results that come back next week.

---

## How It Works — The Architecture

The system is built in three layers: a trained AI model, a backend API, and a user-facing web app.

![Supply Chain Late-Delivery Risk — MLOps Architecture (Data to Deployment)](<asset/Screenshot 2026-05-21 at 12.07.51 PM.png>)

### Layer 1 — The ML Model

**Algorithm:** LightGBM (a fast, accurate decision tree model widely used in industry)

**Training data:** 180,000 real historical supply chain orders from 2015–2016

**What it learned:** Patterns like — "when a Standard Class shipment to West Africa has a short scheduled window and a high discount, it tends to arrive late." Hundreds of such patterns, combined.

**What information it uses to make a prediction:**

| Category | Examples |
|---|---|
| Logistics | Shipping mode (Standard / First Class / etc.), scheduled days |
| Order details | Price, quantity, discount, profit margin |
| Geography | Latitude, longitude, destination region |
| Customer | Segment (Consumer / Corporate), payment type, market |
| Date | Month, day of week, quarter (seasonality matters) |

**How it decides:** The model outputs a number between 0 and 1 — the probability of late delivery. If that number is above **0.35** (the decision threshold), the order is flagged as at risk. The threshold is set low deliberately: it is better to flag a shipment that turns out fine than to miss one that ends up late.

**Model performance:**

| Metric | Score | Plain English |
|---|---|---|
| Recall | 73.2% | Out of every 10 genuinely late shipments, the model catches ~7 |
| Precision | 65.3% | Out of every 10 flagged shipments, ~6–7 are genuinely at risk |
| PR-AUC | 0.807 | Strong overall discrimination between late and on-time |

Recall is the primary metric because **missing a late shipment costs more than a false alarm**. A false alarm costs the ops team a few minutes of extra attention. A missed late shipment costs a customer relationship.

### Layer 2 — The Backend API

Built with **FastAPI** (Python), deployed on **Render**.

The API is the bridge between the web app and the model. It receives an order, runs it through the same preprocessing the model was trained on, scores it, and returns a result. It exposes three endpoints:

| Endpoint | What it does |
|---|---|
| `GET /` | Project overview |
| `GET /health` | Confirms the model is loaded and ready |
| `POST /predict` | Scores a single order, returns probability + flag |
| `POST /predict/batch` | Scores up to 1000 orders at once |

### Layer 3 — The Web App

Built with **Next.js** (React), deployed on **Vercel**.

A clean interface where anyone — no coding required — can fill in the details of an order and instantly see:
- The predicted probability of late delivery (e.g. 62%)
- Whether it is flagged as at risk
- A recommended action ("Flag shipment for extra care before dispatch")

---

## The ML Pipeline (for developers)

Behind the scenes, the model was not simply trained once and forgotten. It is built around three automated pipelines that keep the system healthy over time.

```
Training Pipeline
    Load 180K orders → Validate data quality → Engineer features
    → Train LightGBM → Evaluate (recall, precision, slice checks)
    → Gate: if metrics fail, pipeline stops — bad model never ships
    → Register model in MLflow

Inference Pipeline
    Load new orders → Apply frozen preprocessing → Score with model
    → Write predictions to CSV / return via API

Drift Detection Pipeline
    Compare current order distributions vs training baseline
    → Alert if the data the model sees today looks different
      from what it was trained on
```

**Why drift detection matters:** Imagine the company starts shipping to a new region the model has never seen. The model will still make predictions, but they may be wrong — it has no experience with that region. Drift detection catches this early and alerts the team to retrain.

---

## Project Structure

```
supply-chain-ml-system/
├── api/
│   ├── main.py              # FastAPI app — all endpoints
│   └── schemas.py           # Input/output data shapes
├── configs/
│   └── training_config.yaml # All thresholds, features, hyperparameters
├── core/                    # Pure business logic — no framework dependencies
│   ├── preprocessing.py     # Feature engineering
│   ├── validation.py        # Data quality checks
│   └── evaluation.py        # Metrics and promotion gates
├── pipelines/               # Training, inference, drift orchestration
├── steps/                   # Individual pipeline steps
├── tests/                   # 59 automated tests, run in ~2 seconds
├── frontend/                # Next.js web app
│   ├── app/                 # Pages and layout
│   ├── components/          # PredictionForm, ResultCard, ApiStatus
│   └── lib/api.ts           # Typed API client
├── model/
│   └── pipeline.pkl         # Trained model artifact (frozen)
├── run_training.py          # Entry point: train the model
├── run_inference.py         # Entry point: score new orders
└── run_drift_detection.py   # Entry point: check for data drift
```

---

## Running Locally

### Prerequisites

- Python 3.11
- Node.js 20+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)

### Backend — ML Pipelines

```bash
# Install Python dependencies
uv sync --python 3.11

# First time only: set up ZenML experiment tracking
zenml init
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-registry register mlflow_registry --flavor=mlflow
zenml data-validator register evidently_validator --flavor=evidently
zenml stack register supply-chain-stack \
  -o default -a default \
  -e mlflow_tracker -r mlflow_registry -dv evidently_validator
zenml stack set supply-chain-stack
zenml integration install mlflow evidently -y --uv

# Train the model (reads data/supply_chain.csv → saves model/pipeline.pkl)
uv run python run_training.py

# Score new orders (reads CSVs → writes outputs/predictions_*.csv)
uv run python run_inference.py

# Check for data drift
uv run python run_drift_detection.py
```

### Backend — API Server

```bash
cd api
pip install -r requirements.txt
uvicorn api.main:app --reload
# Visit http://localhost:8000/docs for interactive API documentation
```

### Frontend — Web App

```bash
cd frontend
npm install
npm run dev
# Visit http://localhost:3000
```

### Tests

```bash
uv run pytest tests/ -v
# 59 tests, ~2 seconds, no internet or data files required
```

---

## Configuration

All tunable settings live in `configs/training_config.yaml`. No need to touch Python code to change thresholds or hyperparameters.

| Setting | Default | What it controls |
|---|---|---|
| `model.prediction_threshold` | `0.35` | Above this probability → order is flagged as at risk |
| `evaluation.min_recall` | `0.70` | If the trained model misses more than 30% of late shipments, it is rejected |
| `evaluation.min_precision` | `0.65` | If more than 35% of flags are false alarms, it is rejected |
| `monitoring.drift_share_warning` | `0.30` | Alert if 30%+ of input features look different from training data |
| `data.train_cutoff_date` | `2017-01-01` | Orders before this date train the model; orders after test it |

---

## Deployment

| Component | Platform | URL |
|---|---|---|
| ML Backend API | Render (Docker) | https://late-delivery-prediction-at-the-time-of.onrender.com |
| Web Frontend | Vercel | https://late-delivey-prediction.vercel.app |

The backend is containerised with Docker so it runs identically on any machine or cloud platform. The frontend deploys automatically when code is pushed to the `main` branch.

---

## Conclusion

Before this system existed, late deliveries were discovered after the fact — a frustrating, reactive loop of customer complaints and damage control.

This project replaces that loop with a simple, automated question asked at the moment every order is placed: *Is this shipment at risk?* When the answer is yes, the operations team knows immediately and can act.

The result is a system that:
- **Catches 7 in 10 late shipments** before they leave the warehouse
- **Answers in under a second** via an API any tool can call
- **Never silently degrades** — drift detection and promotion gates ensure only good models serve predictions
- **Requires no ML knowledge to use** — the web interface is as simple as filling in a form

The technology — LightGBM, FastAPI, Next.js, ZenML, MLflow — is production-grade and used at scale in industry. But the goal is simple: fewer late deliveries, fewer frustrated customers, and an operations team that is always one step ahead.

