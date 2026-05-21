import { Suspense } from "react";
import ApiStatus from "@/components/ApiStatus";
import PredictionForm from "@/components/PredictionForm";

export default function Home() {
  return (
    <main className="flex-1 w-full max-w-4xl mx-auto px-4 py-12 sm:px-6 lg:px-8">

      {/* Header */}
      <div className="mb-10">
        <div className="mb-3 flex items-center gap-3">
          <span className="text-2xl">📦</span>
          <Suspense fallback={<span className="text-sm text-slate-500">Checking API…</span>}>
            <ApiStatus />
          </Suspense>
        </div>

        <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
          Late Delivery Prediction
        </h1>
        <p className="mt-3 max-w-2xl text-slate-400">
          Predict whether a shipment will arrive late{" "}
          <span className="text-slate-300 font-medium">at the moment the order is placed</span>.
          Built with LightGBM trained on 125K historical supply chain orders, optimised for high recall
          so at-risk shipments are rarely missed.
        </p>

        {/* Model stats */}
        <div className="mt-6 flex flex-wrap gap-3">
          {[
            { label: "Algorithm", value: "LightGBM" },
            { label: "Training data", value: "125K orders" },
            { label: "Recall", value: "73.2%" },
            { label: "Precision", value: "65.3%" },
            { label: "Threshold", value: "0.35" },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-1.5 text-sm"
            >
              <span className="text-slate-500">{label}: </span>
              <span className="font-medium text-slate-200">{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Prediction form */}
      <PredictionForm />

      {/* Footer */}
      <footer className="mt-12 border-t border-slate-800 pt-6 text-center text-xs text-slate-600">
        Backend API on{" "}
        <a
          href="https://late-delivery-prediction-at-the-time-of.onrender.com/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-500 underline hover:text-slate-300 transition"
        >
          Render
        </a>
        {" · "}
        <a
          href="https://late-delivery-prediction-at-the-time-of.onrender.com/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-slate-500 underline hover:text-slate-300 transition"
        >
          API Docs
        </a>
      </footer>
    </main>
  );
}
