"""Ranking, feasibility and demand scenarios for the combined (all products) view.

Paper link: Section 9.5 - Table 12 (capacity, score, rank, feasibility) and Section 9.6 -
Table 13 (scenario analysis with the revised hybrid D_h x (1 + alpha x g_c)).
Needs results/lstm_seed_stability.json (script 02) and results/survey_results.json (script 01).
Output: results/scenario_analysis.json
"""
import json
import common  # noqa: F401  (adds the repository root to sys.path)
import Logic as L
from common import RES_DIR, load_orders, save_json
from hybrid_patch import compute_hybrid_demand_v2

ALPHA = 0.4
seeds = json.load(open(RES_DIR / "lstm_seed_stability.json"))["seed_stability"]["None"]
g_c = json.load(open(RES_DIR / "survey_results.json"))["g_c_percent"]["all"] / 100.0
suppliers = L.derive_supplier_table_from_raw(load_orders(), None)
full, _ = L.compute_ranking(suppliers, 0.0, L.default_weights())       # scores do not depend on demand
table = full[["Supplier", "Capacity", "Score"]].sort_values("Score", ascending=False).reset_index(drop=True)
table["Rank"] = table.index + 1
table12 = [{"supplier": r.Supplier, "capacity": int(r.Capacity), "score": round(float(r.Score), 3), "rank": int(r.Rank),
            "feasible_if_Dh_at_most": int(r.Capacity),
            "feasible_in_every_seed_run": bool(r.Capacity >= seeds["max"]),
            "feasible_in_no_seed_run": bool(r.Capacity < seeds["min"])} for r in table.itertuples()]
base = compute_hybrid_demand_v2(seeds["median"], g_c, ALPHA, 1.0)
table13 = []
for name, mult in [("Low (x0.8)", 0.8), ("Normal (x1.0)", 1.0), ("High (x1.2)", 1.2)]:
    demand = base * mult
    feasible = [r.Supplier for r in table.itertuples() if r.Capacity >= demand]
    table13.append({"scenario": name, "demand": round(demand), "feasible_in_rank_order": feasible,
                    "recommendation": feasible[0] if feasible else "No feasible supplier"})
out = {"alpha": ALPHA, "g_c": g_c, "D_h_median": seeds["median"], "D_h_min": seeds["min"], "D_h_max": seeds["max"],
       "D_hybrid_normal": round(base), "table12": table12, "table13": table13}
save_json(out, "scenario_analysis.json")
for row in table12: print(row)
for row in table13: print(row)
