import { fetchHealth } from "@/lib/api";

export default async function ApiStatus() {
  let online = false;
  let threshold = 0;

  try {
    const h = await fetchHealth();
    online = h.status === "ok" && h.model_loaded;
    threshold = h.threshold;
  } catch {
    online = false;
  }

  return (
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`inline-block h-2 w-2 rounded-full ${
          online ? "bg-emerald-400" : "bg-red-400"
        }`}
      />
      <span className="text-slate-400">
        {online
          ? `API online · threshold ${threshold}`
          : "API unreachable"}
      </span>
    </div>
  );
}
