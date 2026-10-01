"""Comparison of the prototype's weighted score with SAW and TOPSIS, plus weight sensitivity.

Paper link: Section 10.3, Table 13 (ranks, Kendall tau, entropy weights) and the
Monte Carlo weight-sensitivity result.
Output: results/mcdm_comparison.json
"""
import os
import warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import common  # noqa: F401  (adds the repository root to sys.path)
import Logic as L
from common import load_orders, save_json

suppliers = L.derive_supplier_table_from_raw(load_orders(), None)      # combined (all products) view
names = suppliers["Supplier"].values
crit = pd.DataFrame({
    "Cost": (suppliers["Unit Price"] + suppliers["Transport Cost"]).values,
    "Quality": suppliers["Quality %"].values, "Lead": suppliers["Lead Time (d)"].values,
    "Delay": suppliers["Delay Rate %"].values, "Reliability": suppliers["Reliability %"].values,
    "Capacity": suppliers["Capacity"].values.astype(float)}, index=names)
benefit = {"Cost": 0, "Quality": 1, "Lead": 0, "Delay": 0, "Reliability": 1, "Capacity": 1}

def saw(w):
    z = pd.DataFrame({c: (crit[c] - crit[c].min()) / (crit[c].max() - crit[c].min()) for c in crit})
    for c in crit:
        if not benefit[c]:
            z[c] = 1 - z[c]
    return (z * w).sum(axis=1)

def topsis(w):
    v = crit / np.sqrt((crit ** 2).sum()) * w
    best = pd.Series({c: v[c].max() if benefit[c] else v[c].min() for c in crit})
    worst = pd.Series({c: v[c].min() if benefit[c] else v[c].max() for c in crit})
    d_best, d_worst = np.sqrt(((v - best) ** 2).sum(axis=1)), np.sqrt(((v - worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)

def entropy_weights():
    p = crit / crit.sum()
    d = 1 - (-(p * np.log(p)).sum() / np.log(len(crit)))
    return d / d.sum()

equal = pd.Series(1 / 6, index=crit.columns)
full, _ = L.compute_ranking(suppliers, 0, L.default_weights())
scores = {"Prototype weighted score (5 weights)": pd.Series(full["Score"].values, index=names),
          "SAW (6 equal weights)": saw(equal), "TOPSIS (6 equal weights)": topsis(equal),
          "SAW (entropy weights)": saw(entropy_weights())}
ranks = pd.DataFrame({k: v.rank(ascending=False).astype(int) for k, v in scores.items()})
first = ranks.columns[0]
R = {"scores": {k: v.round(3).to_dict() for k, v in scores.items()}, "ranks": ranks.to_dict(),
     "kendall_tau_vs_prototype": {k: round(float(kendalltau(ranks[first], ranks[k])[0]), 2) for k in ranks.columns[1:]},
     "entropy_weights": entropy_weights().round(3).to_dict()}
rng, top1, draws = np.random.default_rng(0), pd.Series(0, index=names), 5000
for _ in range(draws):
    w = dict(zip(L.WEIGHT_KEYS, rng.dirichlet(np.ones(5))))
    f, _ = L.compute_ranking(suppliers, 0, w)
    top1[f.loc[f["Score"].idxmax(), "Supplier"]] += 1
R["weight_monte_carlo_top1_share"] = (top1 / draws).round(3).to_dict()
save_json(R, "mcdm_comparison.json")
print(ranks); print(R["kendall_tau_vs_prototype"], R["weight_monte_carlo_top1_share"])
