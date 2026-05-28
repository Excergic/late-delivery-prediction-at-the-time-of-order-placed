"""
Supply Chain Late Delivery Prediction API.

Wraps the trained LightGBM pipeline. Accepts raw order data,
applies the same feature engineering as training, and returns
a late delivery risk score and flag.

Security:
    - API key authentication via Authorization: Bearer <key>
      (set API_KEY env var; leave unset to disable for local dev)
    - CORS restricted to frontend origin (FRONTEND_ORIGIN env var)
    - Rate limiting on /predict and /predict/batch (60 req/min per IP)
    - /health endpoint is public (for monitoring probes)

Endpoints:
    GET  /                — project overview and available endpoints
    GET  /health          — liveness check (no auth required)
    POST /predict         — score a single order
    POST /predict/batch   — score up to 1000 orders
"""

from __future__ import annotations

import os
import time
import yaml
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import JSONResponse

from api.schemas import (
    OrderInput, PredictionResult,
    BatchInput, BatchResult,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------

# API key. Set via env var in production. When unset, auth is disabled
# for local development.
API_KEY = os.environ.get("API_KEY", "")

# Frontend origin for CORS. Override for local development:
#   export FRONTEND_ORIGIN="http://localhost:3000"
FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "https://late-delivey-prediction.vercel.app",
)

# Rate limiting: max POST requests per IP per window on prediction endpoints
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "60"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))  # seconds

# ---------------------------------------------------------------------------
# Auth dependency — applies to /predict and /predict/batch
# ---------------------------------------------------------------------------

bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> None:
    """
    Dependency that protects prediction endpoints.

    If API_KEY is set, the request must include a valid
    Authorization: Bearer <key> header. If unset, all requests pass
    through (local development mode).
    """
    if not API_KEY:
        return
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use: Bearer <API_KEY>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Rate limiting — in-memory sliding window per client IP
# ---------------------------------------------------------------------------

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ("/predict", "/predict/batch") and request.method == "POST":
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        cutoff = now - RATE_LIMIT_WINDOW

        timestamps = _rate_limit_store[ip]
        _rate_limit_store[ip] = [t for t in timestamps if t > cutoff]

        if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        _rate_limit_store[ip].append(now)

    return await call_next(request)


# ---------------------------------------------------------------------------
# Config & model paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "model" / "pipeline.pkl"
CONFIG_PATH = BASE_DIR / "configs" / "training_config.yaml"

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

    yield

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

app.middleware("http")(rate_limit_middleware)

# CORS — restrict to the frontend domain only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Feature engineering — identical to training (core/preprocessing.py)
# ---------------------------------------------------------------------------

def _order_to_dataframe(order: OrderInput) -> pd.DataFrame:
    date = pd.to_datetime(order.order_date, errors="coerce")
    if pd.isna(date):
        raise HTTPException(
            status_code=422,
            detail=f"Cannot parse order_date: '{order.order_date}'. "
                   "Use ISO format (2018-01-15) or M/D/YYYY HH:MM.",
        )

    row = {
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
        "order_month":        float(date.month),
        "order_day_of_week":  float(date.dayofweek),
        "order_is_weekend":   float(int(date.dayofweek >= 5)),
        "order_quarter":      float(date.quarter),
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

@app.get("/", tags=["System"])
def root():
    return {
        "title": "Supply Chain Late Delivery Prediction",
        "description": (
            "Predicts which shipments are at risk of arriving late at the time of order placement. "
            "Built with LightGBM + scikit-learn, trained on 125K historical supply chain orders."
        ),
        "version": "1.0.0",
        "endpoints": {
            "GET  /": "Project overview (this response)",
            "GET  /health": "Liveness check — confirms the model is loaded and ready",
            "POST /predict": "Score a single order for late delivery risk (auth required)",
            "POST /predict/batch": "Score up to 1000 orders in one request (auth required)",
            "GET  /docs": "Interactive Swagger UI",
            "GET  /redoc": "ReDoc API documentation",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_pipeline is not None,
        threshold=_threshold or 0.0,
    )


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(order: OrderInput, auth: None = Depends(verify_api_key)):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = _order_to_dataframe(order)
    y_prob, y_pred = _predict_dataframe(df)
    return _make_result(y_prob[0], y_pred[0])


@app.post("/predict/batch", response_model=BatchResult, tags=["Prediction"])
def predict_batch(body: BatchInput, auth: None = Depends(verify_api_key)):
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
