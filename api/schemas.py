"""
Pydantic schemas for the prediction API.

Field names are snake_case for API consumers. Internally they are mapped
to the exact column names the trained sklearn Pipeline expects.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    """A single supply chain order to score."""

    # Date — used to derive order_month, order_day_of_week, order_is_weekend, order_quarter
    order_date: str = Field(
        ...,
        description="Order date. Accepts ISO format (2018-01-15) or M/D/YYYY HH:MM",
        examples=["2018-01-15", "1/15/2018 10:00"],
    )

    # Logistics
    shipping_mode: Literal["Standard Class", "First Class", "Second Class", "Same Day"] = Field(
        ..., description="Shipping mode selected at order time"
    )
    days_for_shipment_scheduled: int = Field(
        ..., ge=0, description="Scheduled number of days to ship"
    )

    # Order financials
    order_item_discount: float = Field(..., ge=0)
    order_item_discount_rate: float = Field(..., ge=0, le=1)
    order_item_product_price: float = Field(..., gt=0)
    order_item_profit_ratio: float = Field(...)
    order_item_quantity: int = Field(..., ge=1)
    sales: float = Field(..., ge=0)
    order_item_total: float = Field(..., ge=0)
    order_profit_per_order: float = Field(...)
    product_price: float = Field(..., gt=0)

    # Geography
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    # Customer / product taxonomy
    payment_type: Literal["DEBIT", "TRANSFER", "CASH", "PAYMENT"] = Field(
        ..., description="Payment type"
    )
    customer_segment: Literal["Consumer", "Corporate", "Home Office"] = Field(...)
    market: Literal["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"] = Field(...)
    department_name: str = Field(..., description="Store department name")
    category_name: str = Field(..., description="Product category name")
    order_region: str = Field(..., description="Geographic region of the order")

    model_config = {
        "json_schema_extra": {
            "example": {
                "order_date": "2018-01-15",
                "shipping_mode": "Standard Class",
                "days_for_shipment_scheduled": 4,
                "order_item_discount": 0.0,
                "order_item_discount_rate": 0.0,
                "order_item_product_price": 49.99,
                "order_item_profit_ratio": 0.25,
                "order_item_quantity": 2,
                "sales": 99.98,
                "order_item_total": 99.98,
                "order_profit_per_order": 24.99,
                "product_price": 49.99,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "payment_type": "DEBIT",
                "customer_segment": "Consumer",
                "market": "USCA",
                "department_name": "Fan Shop",
                "category_name": "Cleats",
                "order_region": "US Northeast",
            }
        }
    }


class PredictionResult(BaseModel):
    """Prediction for a single order."""

    late_delivery_probability: float = Field(
        ..., description="Model's estimated probability of late delivery (0–1)"
    )
    late_delivery_predicted: bool = Field(
        ..., description="True if the order is flagged as at-risk"
    )
    threshold: float = Field(
        ..., description="Decision threshold used to derive the flag"
    )
    recommended_action: str = Field(
        ..., description="Suggested ops team action based on the prediction"
    )


class BatchInput(BaseModel):
    orders: list[OrderInput] = Field(..., min_length=1, max_length=1000)


class BatchResult(BaseModel):
    predictions: list[PredictionResult]
    flagged_count: int
    total_count: int
    flag_rate: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float
