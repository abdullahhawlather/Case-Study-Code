"""Holt damped-trend forecast diagnostics (legacy filename retained).

This script replaces the former LSTM random-seed experiment. Holt damped-trend
smoothing is deterministic, so neural-network seed stability is no longer relevant.
It computes the 30-day historical-demand forecast for the combined series and each
product, using the same prototype function as Logic.py.

Usage: python 02_lstm_seed_stability.py
Output: results/holt_forecast_diagnostics.json
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import common  # noqa: F401
import Logic as L
from common import load_history, save_json

HORIZON_DAYS = 30
hist = load_history()
out = {}
for pid in [None, "P001", "P002", "P003", "P004", "P005"]:
    h = hist.copy() if pid is None else hist[hist["Product ID"].astype(str) == pid].copy()
    if pid is None:
        h = L.aggregate_historical_across_products(h)
    h = L.parse_historical_upload(h)
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")
    h["Demand"] = pd.to_numeric(h["Demand"], errors="coerce")
    h = h.dropna(subset=["Date", "Demand"]).sort_values("Date")
    forecast, used_history, avg_step = L.compute_historical_forecast(h, HORIZON_DAYS)
    out[str(pid)] = {
        "model": "Holt damped-trend exponential smoothing",
        "horizon_days": HORIZON_DAYS,
        "n_observations": int(len(h)),
        "used_history": bool(used_history),
        "average_observation_step_days": float(avg_step) if avg_step is not None else None,
        "forecast": float(forecast),
    }
    print(f"{pid}: Holt forecast={forecast:.2f}; observations={len(h)}")

save_json({"model": "Holt damped-trend exponential smoothing", "diagnostics": out},
          "holt_forecast_diagnostics.json")
