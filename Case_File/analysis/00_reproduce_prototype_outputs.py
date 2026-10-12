"""Re-run the prototype logic (Logic.py) for every product and the combined view.

Paper link: Section 9.5 (Tables 8 and 9) and Section 9.6 (consumer-based demand D_c,
original additive hybrid at alpha = 0.4).
Output: results/prototype_outputs.json
"""
import os
import warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import pandas as pd
import common  # noqa: F401  (adds the repository root to sys.path)
import Logic as L
from common import load_orders, load_history, SURVEY_PROTO_FILE, save_json

orders, hist, survey = load_orders(), load_history(), pd.read_excel(SURVEY_PROTO_FILE)
results = {}
for pid in [None, "P001", "P002", "P003", "P004", "P005"]:
    h = hist if pid is None else hist[hist["Product ID"] == pid]
    s = survey if pid is None else survey[survey["Product ID"] == pid]
    if pid is None:
        h = L.aggregate_historical_across_products(h)
    h = L.parse_historical_upload(h)
    consumer = L.parse_consumer_upload(s)
    d_h, _, _ = L.compute_historical_forecast(h, 30)
    d_c = L.compute_consumer_demand(consumer)
    suppliers = L.derive_supplier_table_from_raw(orders, pid)
    weights = L.default_weights()
    rec = {"n_history": len(h), "n_survey": len(s), "D_h": float(d_h), "D_c": float(d_c)}
    for alpha in (0.0, 0.4):
        demand = L.compute_hybrid_demand(d_c, d_h, alpha, 1.0)
        full, ranked = L.compute_ranking(suppliers, demand, weights)
        rec[f"alpha_{alpha}"] = {
            "hybrid_demand": float(demand),
            "feasible_suppliers": int(full["Feasible"].sum()),
            "ranking": ranked[["Supplier", "Score"]].round(3).values.tolist(),
        }
    results[str(pid)] = rec
    print(pid, {k: v for k, v in rec.items() if not k.startswith("alpha")})
save_json(results, "prototype_outputs.json")
