"use client";

import { PredictionResult } from "@/lib/api";

interface Props {
  result: PredictionResult;
}

export default function ResultCard({ result }: Props) {
  const { late_delivery_probability: prob, late_delivery_predicted: flagged, recommended_action } = result;
  const pct = Math.round(prob * 100);

  const riskLevel = prob >= 0.65 ? "high" : prob >= 0.45 ? "medium" : "low";
  const riskConfig = {
    high:   { label: "High Risk",   bar: "bg-red-500",    badge: "bg-red-500/15 text-red-400 ring-red-500/30" },
    medium: { label: "Medium Risk", bar: "bg-amber-500",  badge: "bg-amber-500/15 text-amber-400 ring-amber-500/30" },
    low:    { label: "Low Risk",    bar: "bg-emerald-500",badge: "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30" },
  }[riskLevel];

  return (
    <div className="mt-8 rounded-2xl border border-slate-700 bg-slate-800/60 p-6 backdrop-blur">
      <h3 className="mb-6 text-lg font-semibold text-white">Prediction Result</h3>

      {/* Top metrics */}
      <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div className="rounded-xl bg-slate-900/60 p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Probability</p>
          <p className="mt-1 text-3xl font-bold text-white">{pct}%</p>
        </div>
        <div className="rounded-xl bg-slate-900/60 p-4">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Risk Level</p>
          <span
            className={`mt-2 inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ring-1 ring-inset ${riskConfig.badge}`}
          >
            {riskConfig.label}
          </span>
        </div>
        <div className="rounded-xl bg-slate-900/60 p-4 col-span-2 sm:col-span-1">
          <p className="text-xs text-slate-400 uppercase tracking-wide">Flagged</p>
          <p className="mt-1 text-2xl font-bold text-white">{flagged ? "Yes" : "No"}</p>
        </div>
      </div>

      {/* Probability bar */}
      <div className="mb-6">
        <div className="mb-1 flex justify-between text-xs text-slate-400">
          <span>0%</span>
          <span>Late delivery probability</span>
          <span>100%</span>
        </div>
        <div className="h-3 w-full overflow-hidden rounded-full bg-slate-700">
          <div
            className={`h-full rounded-full transition-all duration-700 ${riskConfig.bar}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        {/* Threshold marker */}
        <div className="relative mt-1 h-2" style={{ marginLeft: `${result.threshold * 100}%` }}>
          <span className="absolute -translate-x-1/2 text-[10px] text-slate-500">
            ▲ threshold ({Math.round(result.threshold * 100)}%)
          </span>
        </div>
      </div>

      {/* Recommended action */}
      <div
        className={`rounded-xl p-4 ring-1 ring-inset ${
          flagged
            ? "bg-red-500/10 ring-red-500/20 text-red-300"
            : "bg-emerald-500/10 ring-emerald-500/20 text-emerald-300"
        }`}
      >
        <p className="text-sm font-medium">{recommended_action}</p>
      </div>
    </div>
  );
}
