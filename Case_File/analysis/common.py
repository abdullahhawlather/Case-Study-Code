"""Shared paths, loaders, and forecasting helpers for the analysis scripts.

The historical-demand model is Holt's damped-trend exponential smoothing. The
file name ``02_lstm_seed_stability.py`` is retained only for compatibility with
the user's existing project; its contents no longer train or evaluate an LSTM.
"""
import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
# Files may live either in the repository root or in an analysis/ subfolder.
REPO_ROOT = ANALYSIS_DIR if (ANALYSIS_DIR / "Logic.py").exists() else ANALYSIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _first_existing(*paths):
    candidates = [Path(path) for path in paths if path is not None]
    for path in candidates:
        if path.exists():
            return path
    # Return the final candidate so the resulting error points to the most local path.
    return candidates[-1] if candidates else ANALYSIS_DIR


# Support both the original uploaded folder structure and a flat project directory.
_proto_dir = _first_existing(REPO_ROOT / "Prototype Upload Data", ANALYSIS_DIR / "Prototype Upload Data")
DATA_DIR = _proto_dir if _proto_dir.exists() else REPO_ROOT
_main_dir = _first_existing(DATA_DIR / "Main Dataset", REPO_ROOT / "Main Dataset")
MAIN_DIR = _main_dir if _main_dir.exists() else DATA_DIR
FIG_DIR = ANALYSIS_DIR / "figures"
RES_DIR = ANALYSIS_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

ORDERS_FILE = _first_existing(
    MAIN_DIR / "Purchase_Orders.xlsx", DATA_DIR / "Purchase_Orders.xlsx",
    REPO_ROOT / "Purchase_Orders.xlsx", ANALYSIS_DIR / "Purchase_Orders.xlsx")
HISTORY_FILE = _first_existing(
    DATA_DIR / "Historical Data.xlsx", REPO_ROOT / "Historical Data.xlsx",
    ANALYSIS_DIR / "Historical Data.xlsx")
SURVEY_PROTO_FILE = _first_existing(
    DATA_DIR / "Customer_Survey_Final.xlsx", REPO_ROOT / "Customer_Survey_Final.xlsx",
    ANALYSIS_DIR / "Customer_Survey_Final.xlsx")
SURVEY_ANON_FILE = _first_existing(
    MAIN_DIR / "Survey_Responses_Anonymised.xlsx", DATA_DIR / "Survey_Responses_Anonymised.xlsx",
    REPO_ROOT / "Survey_Responses_Anonymised.xlsx", ANALYSIS_DIR / "Survey_Responses_Anonymised.xlsx")
SURVEY_ORIG_FILE = _first_existing(
    MAIN_DIR / "Survey_For_Prototype_(Responses)_Original.xlsx",
    DATA_DIR / "Survey_For_Prototype_(Responses)_Original.xlsx",
    REPO_ROOT / "Survey_For_Prototype_(Responses)_Original.xlsx",
    ANALYSIS_DIR / "Survey_For_Prototype_(Responses)_Original.xlsx")
SUPPLIER_FILE = _first_existing(
    DATA_DIR / "Supplier Data.xlsx", REPO_ROOT / "Supplier Data.xlsx", ANALYSIS_DIR / "Supplier Data.xlsx")


def save_json(obj, name):
    path = RES_DIR / name
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str, allow_nan=False)
    print("saved", path)
    return path


def load_orders():
    import pandas as pd
    if not ORDERS_FILE.exists():
        raise FileNotFoundError(f"Purchase-orders workbook not found. Looked for: {ORDERS_FILE}")
    return pd.read_excel(ORDERS_FILE, sheet_name="Sales_data")


def load_history():
    import pandas as pd
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"Historical-demand workbook not found. Looked for: {HISTORY_FILE}")
    return pd.read_excel(HISTORY_FILE, sheet_name="Historical Demand")


def holt_damped_forecast(series, steps):
    """Return a non-negative Holt damped-trend forecast for ``steps`` periods.

    Falls back to a last-value forecast for short or constant series. This is a
    deterministic statistical model; no random seed or neural-network training
    is involved.
    """
    import numpy as np
    from statsmodels.tsa.holtwinters import Holt

    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    steps = max(1, int(steps))
    if len(y) == 0:
        return np.zeros(steps, dtype=float)
    if len(y) < 3 or np.ptp(y) < 1e-12:
        return np.full(steps, max(0.0, float(y[-1])), dtype=float)
    try:
        fit = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
        forecast = np.asarray(fit.forecast(steps), dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        forecast = np.full(steps, float(y[-1]), dtype=float)
    forecast = np.where(np.isfinite(forecast), forecast, float(y[-1]))
    return np.clip(forecast, 0.0, None)
