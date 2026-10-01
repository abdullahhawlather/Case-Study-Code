"""Run-to-run variability of the prototype LSTM forecast.

Paper link: Section 9.5, Table 8 (median and range of the 30-day-horizon output over
10 random seeds) and Table 9 (feasibility range in the combined view).
Usage: python 02_lstm_seed_stability.py [--seeds 10]
Output: results/lstm_seed_stability.json
"""
import argparse
import os
import warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import common  # noqa: F401  (adds the repository root to sys.path)
import Logic as L
from common import load_history, lstm_forecast, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=10)
n_seeds = ap.parse_args().seeds

hist, out = load_history(), {}
for pid in [None, "P001", "P002", "P003", "P004", "P005"]:
    h = hist if pid is None else hist[hist["Product ID"] == pid]
    if pid is None:
        h = L.aggregate_historical_across_products(h)
    h = L.parse_historical_upload(h)
    h = h.assign(Date=pd.to_datetime(h["Date"])).sort_values("Date")
    steps = int(30 / 7)                       # as in Logic.compute_historical_forecast
    vals = [float(lstm_forecast(h["Demand"].values, steps, seed)[-1]) for seed in range(n_seeds)]
    out[str(pid)] = {"n_observations": len(h), "median": float(np.median(vals)),
                     "min": float(min(vals)), "max": float(max(vals)), "all_runs": vals}
    print(pid, round(out[str(pid)]["median"]), round(out[str(pid)]["min"]), round(out[str(pid)]["max"]))
save_json({"seed_stability": out}, "lstm_seed_stability.json")
