"use client";

import { useState } from "react";
import { predict, PredictionResult, OrderInput } from "@/lib/api";
import ResultCard from "@/components/ResultCard";

const SHIPPING_MODES = ["Standard Class", "First Class", "Second Class", "Same Day"] as const;
const PAYMENT_TYPES  = ["DEBIT", "TRANSFER", "CASH", "PAYMENT"] as const;
const CUSTOMER_SEGS  = ["Consumer", "Corporate", "Home Office"] as const;
const MARKETS        = ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"] as const;

const DEPARTMENT_NAMES = [
  "Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors", "Technology",
  "Fitness", "Book Shop", "Discs Shop", "Health and Beauty", "Jewelry", "Pets", "Auto Parts",
];
const CATEGORY_NAMES = [
  "Cleats", "Men's Footwear", "Women's Apparel", "Indoor/Outdoor Games",
  "Camping & Hiking", "Water Sports", "Cardio Equipment", "Fishing",
  "Golf Bags & Carts", "Garden", "Cameras", "Computers", "Electronics",
  "Tennis & Racquet", "Basketball", "Soccer",
];
const ORDER_REGIONS = [
  "US Northeast", "US South", "US Southeast", "US Southwest", "US West",
  "Western Europe", "Central America", "Southeast Asia", "Eastern Asia",
  "Caribbean", "Northern Africa", "West Africa", "East Africa",
  "South Asia", "Oceania", "Eastern Europe", "Central Africa", "Southern Africa", "North Africa",
];

const DEFAULTS: OrderInput = {
  order_date: "2018-01-15",
  shipping_mode: "Standard Class",
  days_for_shipment_scheduled: 4,
  order_item_discount: 0,
  order_item_discount_rate: 0,
  order_item_product_price: 49.99,
  order_item_profit_ratio: 0.25,
  order_item_quantity: 2,
  sales: 99.98,
  order_item_total: 99.98,
  order_profit_per_order: 24.99,
  product_price: 49.99,
  latitude: 40.71,
  longitude: -74.01,
  payment_type: "DEBIT",
  customer_segment: "Consumer",
  market: "USCA",
  department_name: "Fan Shop",
  category_name: "Cleats",
  order_region: "US Northeast",
};

// ── Shared field components ──────────────────────────────────────────────────

function Label({ children }: { children: React.ReactNode }) {
  return (
    <label className="mb-1 block text-xs font-medium text-slate-400 uppercase tracking-wide">
      {children}
    </label>
  );
}

const inputCls =
  "w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition";

const selectCls =
  "w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition";

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h4 className="mt-6 mb-3 text-xs font-semibold uppercase tracking-widest text-indigo-400">
      {children}
    </h4>
  );
}

// ── Main form ─────────────────────────────────────────────────────────────────

export default function PredictionForm() {
  const [form, setForm] = useState<OrderInput>(DEFAULTS);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof OrderInput>(key: K, value: OrderInput[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predict(form);
      setResult(res);
      // Scroll result into view
      setTimeout(() => document.getElementById("result")?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="rounded-2xl border border-slate-700 bg-slate-800/60 p-6 backdrop-blur">

      {/* Logistics */}
      <SectionTitle>Logistics</SectionTitle>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div>
          <Label>Order Date</Label>
          <input
            type="date"
            className={inputCls}
            value={form.order_date}
            onChange={(e) => set("order_date", e.target.value)}
            required
          />
        </div>
        <div>
          <Label>Shipping Mode</Label>
          <select
            className={selectCls}
            value={form.shipping_mode}
            onChange={(e) => set("shipping_mode", e.target.value as OrderInput["shipping_mode"])}
          >
            {SHIPPING_MODES.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <Label>Scheduled Shipment Days</Label>
          <input
            type="number"
            min={0}
            max={30}
            className={inputCls}
            value={form.days_for_shipment_scheduled}
            onChange={(e) => set("days_for_shipment_scheduled", parseInt(e.target.value))}
            required
          />
        </div>
      </div>

      {/* Order Financials */}
      <SectionTitle>Order Financials</SectionTitle>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        <div>
          <Label>Product Price ($)</Label>
          <input type="number" step="0.01" min="0.01" className={inputCls}
            value={form.product_price}
            onChange={(e) => set("product_price", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Order Item Product Price ($)</Label>
          <input type="number" step="0.01" min="0.01" className={inputCls}
            value={form.order_item_product_price}
            onChange={(e) => set("order_item_product_price", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Quantity</Label>
          <input type="number" min="1" className={inputCls}
            value={form.order_item_quantity}
            onChange={(e) => set("order_item_quantity", parseInt(e.target.value))} required />
        </div>
        <div>
          <Label>Sales ($)</Label>
          <input type="number" step="0.01" min="0" className={inputCls}
            value={form.sales}
            onChange={(e) => set("sales", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Order Item Total ($)</Label>
          <input type="number" step="0.01" min="0" className={inputCls}
            value={form.order_item_total}
            onChange={(e) => set("order_item_total", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Item Discount ($)</Label>
          <input type="number" step="0.01" min="0" className={inputCls}
            value={form.order_item_discount}
            onChange={(e) => set("order_item_discount", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Discount Rate (0–1)</Label>
          <input type="number" step="0.01" min="0" max="1" className={inputCls}
            value={form.order_item_discount_rate}
            onChange={(e) => set("order_item_discount_rate", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Profit Ratio</Label>
          <input type="number" step="0.01" className={inputCls}
            value={form.order_item_profit_ratio}
            onChange={(e) => set("order_item_profit_ratio", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Profit Per Order ($)</Label>
          <input type="number" step="0.01" className={inputCls}
            value={form.order_profit_per_order}
            onChange={(e) => set("order_profit_per_order", parseFloat(e.target.value))} required />
        </div>
      </div>

      {/* Geography */}
      <SectionTitle>Geography</SectionTitle>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label>Latitude</Label>
          <input type="number" step="0.0001" min="-90" max="90" className={inputCls}
            value={form.latitude}
            onChange={(e) => set("latitude", parseFloat(e.target.value))} required />
        </div>
        <div>
          <Label>Longitude</Label>
          <input type="number" step="0.0001" min="-180" max="180" className={inputCls}
            value={form.longitude}
            onChange={(e) => set("longitude", parseFloat(e.target.value))} required />
        </div>
      </div>

      {/* Customer & Product */}
      <SectionTitle>Customer &amp; Product</SectionTitle>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div>
          <Label>Payment Type</Label>
          <select className={selectCls} value={form.payment_type}
            onChange={(e) => set("payment_type", e.target.value as OrderInput["payment_type"])}>
            {PAYMENT_TYPES.map((t) => <option key={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <Label>Customer Segment</Label>
          <select className={selectCls} value={form.customer_segment}
            onChange={(e) => set("customer_segment", e.target.value as OrderInput["customer_segment"])}>
            {CUSTOMER_SEGS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </div>
        <div>
          <Label>Market</Label>
          <select className={selectCls} value={form.market}
            onChange={(e) => set("market", e.target.value as OrderInput["market"])}>
            {MARKETS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
        <div>
          <Label>Department Name</Label>
          <select className={selectCls} value={form.department_name}
            onChange={(e) => set("department_name", e.target.value)}>
            {DEPARTMENT_NAMES.map((d) => <option key={d}>{d}</option>)}
          </select>
        </div>
        <div>
          <Label>Category Name</Label>
          <select className={selectCls} value={form.category_name}
            onChange={(e) => set("category_name", e.target.value)}>
            {CATEGORY_NAMES.map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <Label>Order Region</Label>
          <select className={selectCls} value={form.order_region}
            onChange={(e) => set("order_region", e.target.value)}>
            {ORDER_REGIONS.map((r) => <option key={r}>{r}</option>)}
          </select>
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading}
        className="mt-8 w-full rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white hover:bg-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition"
      >
        {loading ? "Scoring order…" : "Predict Late Delivery Risk"}
      </button>

      {/* Error */}
      {error && (
        <p className="mt-4 rounded-lg bg-red-500/10 px-4 py-3 text-sm text-red-400 ring-1 ring-red-500/20">
          {error}
        </p>
      )}

      {/* Result */}
      <div id="result">
        {result && <ResultCard result={result} />}
      </div>
    </form>
  );
}
