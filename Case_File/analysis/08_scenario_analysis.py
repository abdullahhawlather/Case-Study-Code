"""Supplier feasibility and demand scenarios using the Holt-based hybrid forecast.

Uses the deterministic 30-day Holt damped-trend forecast from script 02 and the
survey-implied growth statistic from script 01. No LSTM seed range is assumed.
Output: results/scenario_analysis.json
"""
import json
import common  # noqa: F401
import Logic as L
from common import RES_DIR, load_orders, save_json
from hybrid_patch import compute_hybrid_demand_v2

ALPHA = 0.4
forecast_path = RES_DIR / "holt_forecast_diagnostics.json"
survey_path = RES_DIR / "survey_results.json"
if not forecast_path.exists():
    raise FileNotFoundError(f"Run 02_lstm_seed_stability.py first; missing {forecast_path}")
if not survey_path.exists():
    raise FileNotFoundError(f"Run 01_survey_analysis.py first; missing {survey_path}")
with open(forecast_path, encoding="utf-8") as fh:
    diagnostics = json.load(fh)["diagnostics"]
with open(survey_path, encoding="utf-8") as fh:
    survey_results = json.load(fh)

base_historical = float(diagnostics["None"]["forecast"])
g_c = float(survey_results["g_c_percent"]["all"]) / 100.0
suppliers = L.derive_supplier_table_from_raw(load_orders(), None)
full, _ = L.compute_ranking(suppliers, 0.0, L.default_weights())
table = full[["Supplier", "Capacity", "Score"]].sort_values("Score", ascending=False).reset_index(drop=True)
table["Rank"] = table.index + 1
base = compute_hybrid_demand_v2(base_historical, g_c, ALPHA, 1.0)

table12 = []
for r in table.itertuples():
    table12.append({
        "supplier": r.Supplier,
        "capacity": float(r.Capacity),
        "score": round(float(r.Score), 3),
        "rank": int(r.Rank),
        "forecasted_demand": round(base, 2),
        "feasible_at_normal_demand": bool(float(r.Capacity) >= base),
    })

table13 = []
for name, mult in [("Low (x0.8)", 0.8), ("Normal (x1.0)", 1.0), ("High (x1.2)", 1.2)]:
    demand = base * mult
    feasible = [r.Supplier for r in table.itertuples() if float(r.Capacity) >= demand]
    table13.append({
        "scenario": name,
        "scenario_multiplier": mult,
        "demand": round(demand, 2),
        "feasible_in_rank_order": feasible,
        "recommendation": feasible[0] if feasible else "No feasible supplier",
    })

out = {
    "historical_model": "Holt damped-trend exponential smoothing",
    "historical_forecast_30_day": round(base_historical, 2),
    "alpha": ALPHA,
    "survey_implied_growth": g_c,
    "hybrid_formula": "D_hybrid = D_h * (1 + alpha * g_c) * scenario",
    "D_hybrid_normal": round(base, 2),
    "forecast_is_deterministic": True,
    "table12": table12,
    "table13": table13,
}
save_json(out, "scenario_analysis.json")
print("Historical forecast (Holt damped trend):", round(base_historical, 2))
print("Normal hybrid demand:", round(base, 2))
for row in table13:
    print(row)
