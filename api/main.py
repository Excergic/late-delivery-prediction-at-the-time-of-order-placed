"""
Supply Chain Late Delivery Prediction API.

Wraps the trained LightGBM pipeline. Accepts raw order data,
applies the same feature engineering as training, and returns
a late delivery risk score and flag.

Endpoints:
    GET  /health          — liveness check
    POST /predict         — score a single order
    POST /predict/batch   — score up to 1000 orders
"""

from __future__ import annotations

import os
import yaml
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    OrderInput, PredictionResult,
    BatchInput, BatchResult,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Config & model paths — resolved relative to this file so they work both
# locally (python -m uvicorn) and inside a Docker container.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "pipeline.pkl"
CONFIG_PATH = BASE_DIR / "configs" / "training_config.yaml"

# Loaded once at startup
_pipeline = None
_config = None
_threshold = None


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _config, _threshold

    with open(CONFIG_PATH) as f:
        _config = yaml.safe_load(f)
    _threshold = _config["model"]["prediction_threshold"]

    _pipeline = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Decision threshold: {_threshold}")

    yield  # app is running

    _pipeline = None


app = FastAPI(
    title="Supply Chain Late Delivery Prediction",
    description=(
        "Predicts which shipments are at risk of arriving late. "
        "Built with LightGBM + scikit-learn, trained on 125K historical orders. "
        "Primary metric: Recall (missing a late shipment costs more than a false alarm)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Feature engineering — identical logic to training (core/preprocessing.py)
# Duplicated here so the API has zero dependency on the training codebase.
# ---------------------------------------------------------------------------

def _order_to_dataframe(order: OrderInput) -> pd.DataFrame:
    """
    Convert a single OrderInput into a one-row DataFrame with the exact
    column names and date-derived features the trained pipeline expects.
    """
    # Parse date
    date = pd.to_datetime(order.order_date, errors="coerce")
    if pd.isna(date):
        raise HTTPException(
            status_code=422,
            detail=f"Cannot parse order_date: '{order.order_date}'. "
                   "Use ISO format (2018-01-15) or M/D/YYYY HH:MM."
        )

    row = {
        # Numeric features
        "Days for shipment (scheduled)": order.days_for_shipment_scheduled,
        "Order Item Discount":           order.order_item_discount,
        "Order Item Discount Rate":      order.order_item_discount_rate,
        "Order Item Product Price":      order.order_item_product_price,
        "Order Item Profit Ratio":       order.order_item_profit_ratio,
        "Order Item Quantity":           order.order_item_quantity,
        "Sales":                         order.sales,
        "Order Item Total":              order.order_item_total,
        "Order Profit Per Order":        order.order_profit_per_order,
        "Product Price":                 order.product_price,
        "Latitude":                      order.latitude,
        "Longitude":                     order.longitude,
        # Date-derived numeric features
        "order_month":        float(date.month),
        "order_day_of_week":  float(date.dayofweek),
        "order_is_weekend":   float(int(date.dayofweek >= 5)),
        "order_quarter":      float(date.quarter),
        # Categorical features
        "Shipping Mode":      order.shipping_mode,
        "Type":               order.payment_type,
        "Customer Segment":   order.customer_segment,
        "Market":             order.market,
        "Department Name":    order.department_name,
        "Category Name":      order.category_name,
        "Order Region":       order.order_region,
    }
    return pd.DataFrame([row])


def _predict_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the frozen sklearn Pipeline to a feature DataFrame.
    Returns (probabilities, binary_predictions).
    """
    preprocessor = _pipeline.named_steps["preprocessor"]
    model = _pipeline.named_steps["model"]

    X_transformed = pd.DataFrame(preprocessor.transform(df))
    y_prob = model.predict_proba(X_transformed)[:, 1]
    y_pred = (y_prob >= _threshold).astype(int)

    return y_prob, y_pred


def _make_result(prob: float, pred: int) -> PredictionResult:
    flagged = bool(pred)
    return PredictionResult(
        late_delivery_probability=round(float(prob), 4),
        late_delivery_predicted=flagged,
        threshold=_threshold,
        recommended_action=(
            "Flag shipment for extra care before dispatch."
            if flagged
            else "No action required — low late delivery risk."
        ),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check. Returns 200 when the model is loaded and ready."""
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        threshold=_threshold or 0.0,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(order: OrderInput):
    """
    Score a single order and return its late delivery risk.

    Accepts raw order fields (same data available at order placement time —
    no delivery outcome columns). Applies the same feature engineering as
    the training pipeline before scoring.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _order_to_dataframe(order)
    y_prob, y_pred = _predict_dataframe(df)
    return _make_result(y_prob[0], y_pred[0])


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(body: BatchInput):
    """
    Score up to 1000 orders in a single request.

    Returns predictions in the same order as the input list,
    plus summary statistics (flag count, flag rate).
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = [_order_to_dataframe(o) for o in body.orders]
    df = pd.concat(rows, ignore_index=True)
    y_prob, y_pred = _predict_dataframe(df)

    predictions = [_make_result(p, d) for p, d in zip(y_prob, y_pred)]
    flagged = int(y_pred.sum())

    return BatchResult(
        predictions=predictions,
        flagged_count=flagged,
        total_count=len(predictions),
        flag_rate=round(flagged / len(predictions), 4),
    )
