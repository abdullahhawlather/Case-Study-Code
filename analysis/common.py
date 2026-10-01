"""Shared paths and helpers for the analysis scripts.

The scripts in this folder reproduce the tables and figures of the case study
"An AI-Driven Dynamic Supplier Selection Framework with Hybrid Demand Forecasting
and Scenario-Based Decision Support". Place this folder (analysis/) in the root of
the repository, next to Logic.py and App.py.
"""
import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
DATA_DIR = REPO_ROOT / "Prototype Upload Data"
MAIN_DIR = DATA_DIR / "Main Dataset"
FIG_DIR = ANALYSIS_DIR / "figures"
RES_DIR = ANALYSIS_DIR / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

# Make the prototype logic (Logic.py) importable.
sys.path.insert(0, str(REPO_ROOT))

ORDERS_FILE = MAIN_DIR / "Purchase_Orders.xlsx"            # sheet "Sales_data"
HISTORY_FILE = DATA_DIR / "Historical Data.xlsx"
SURVEY_PROTO_FILE = DATA_DIR / "Customer_Survey_Final.xlsx"
SURVEY_ANON_FILE = MAIN_DIR / "Survey_Responses_Anonymised.xlsx"
SURVEY_ORIG_FILE = MAIN_DIR / "Survey_For_Prototype_(Responses)_Original.xlsx"


def save_json(obj, name):
    path = RES_DIR / name
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    print("saved", path)


def load_orders():
    import pandas as pd
    return pd.read_excel(ORDERS_FILE, sheet_name="Sales_data")


def load_history():
    import pandas as pd
    return pd.read_excel(HISTORY_FILE, sheet_name="Sheet1")


def lstm_forecast(series, steps, seed, activation="relu", lookback=3, epochs=25):
    """Prototype LSTM configuration (paper Section 9.3): one LSTM layer (32 units),
    one Dense layer, Adam, MSE, min-max scaling, 3-step window, batch size 2."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import numpy as np
    import tensorflow as tf
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    s = scaler.fit_transform(np.array(series, float).reshape(-1, 1))
    X = np.array([s[i - lookback:i, 0] for i in range(lookback, len(s))]).reshape(-1, lookback, 1)
    y = np.array([s[i, 0] for i in range(lookback, len(s))])
    tf.keras.utils.set_random_seed(seed)
    model = Sequential([LSTM(32, activation=activation, input_shape=(lookback, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=2, verbose=0)
    window = s[-lookback:].reshape(1, lookback, 1)
    out = []
    for _ in range(steps):
        p = model.predict(window, verbose=0)[0][0]
        out.append(p)
        window = np.append(window[0, 1:, 0], p).reshape(1, lookback, 1)
    return scaler.inverse_transform(np.array(out).reshape(-1, 1)).ravel()
