const API_BASE = "https://late-delivery-prediction-at-the-time-of.onrender.com";

export interface OrderInput {
  order_date: string;
  shipping_mode: "Standard Class" | "First Class" | "Second Class" | "Same Day";
  days_for_shipment_scheduled: number;
  order_item_discount: number;
  order_item_discount_rate: number;
  order_item_product_price: number;
  order_item_profit_ratio: number;
  order_item_quantity: number;
  sales: number;
  order_item_total: number;
  order_profit_per_order: number;
  product_price: number;
  latitude: number;
  longitude: number;
  payment_type: "DEBIT" | "TRANSFER" | "CASH" | "PAYMENT";
  customer_segment: "Consumer" | "Corporate" | "Home Office";
  market: "LATAM" | "Europe" | "Pacific Asia" | "USCA" | "Africa";
  department_name: string;
  category_name: string;
  order_region: string;
}

export interface PredictionResult {
  late_delivery_probability: number;
  late_delivery_predicted: boolean;
  threshold: number;
  recommended_action: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  threshold: number;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`, { next: { revalidate: 60 } });
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}

export async function predict(order: OrderInput): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(order),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `API error ${res.status}`);
  }
  return res.json();
}
