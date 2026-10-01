"""Chronological forecast evaluation of weekly total order quantity.

Paper link: Section 10.1, Table 11 (naive, moving average, training mean, exponential
smoothing, ARIMA(1,0,1), LSTM with ReLU and tanh).
Split: first 70% / next 15% / last 15% of weeks; one-step-ahead forecasts.
Usage: python 03_forecast_evaluation.py [--seeds 5]
Output: results/forecast_evaluation.json
"""
import argparse
import os
import warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from common import load_orders, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
n_seeds = ap.parse_args().seeds

orders = load_orders()
orders["od"] = pd.to_datetime(orders["Order_Date"])
weekly = orders.set_index("od")["Quantity"].resample("W-MON").sum()
weekly = weekly[weekly.index < "2024-01-02"]
y = weekly.values.astype(float)
n = len(y)
n_train, n_val = int(n * .7), int(n * .15)
print("weeks:", n, "split:", n_train, n_val, n - n_train - n_val)

def metrics(actual, forecast):
    a, f = np.array(actual), np.array(forecast)
    return {"MAE": float(np.mean(abs(a - f))), "RMSE": float(np.sqrt(np.mean((a - f) ** 2))),
            "MAPE": float(np.mean(abs(a - f) / np.maximum(a, 1)) * 100)}

def statistical_models(lo, hi):
    actual, res = y[lo:hi], {}
    res["Naive (last week)"] = metrics(actual, y[lo - 1:hi - 1])
    res["Moving average (4 wk)"] = metrics(actual, [y[i - 4:i].mean() for i in range(lo, hi)])
    res["Training mean"] = metrics(actual, [y[:lo].mean()] * (hi - lo))
    ses = SimpleExpSmoothing(y[:lo], initialization_method="estimated").fit()
    alpha, level, f = ses.params["smoothing_level"], ses.level[-1], []
    for i in range(lo, hi):
        f.append(level); level = alpha * y[i] + (1 - alpha) * level
    res["Exp. smoothing (SES)"] = metrics(actual, f)
    res["_ses_smoothing_level"] = float(alpha)
    model, f = ARIMA(y[:lo], order=(1, 0, 1)).fit(), []
    for i in range(lo, hi):
        f.append(model.forecast(1)[0]); model = model.append([y[i]], refit=False)
    res["ARIMA(1,0,1)"] = metrics(actual, f)
    return res

def lstm_eval(lo, hi, train_end, activation):
    lookback = 3
    scaler = MinMaxScaler().fit(y[:train_end].reshape(-1, 1))
    s = scaler.transform(y.reshape(-1, 1))[:, 0]
    X = np.array([s[i - lookback:i] for i in range(lookback, train_end)]).reshape(-1, lookback, 1)
    t = s[lookback:train_end]
    runs = []
    for seed in range(n_seeds):
        tf.keras.utils.set_random_seed(seed)
        m = Sequential([LSTM(32, activation=activation, input_shape=(lookback, 1)), Dense(1)])
        m.compile(optimizer="adam", loss="mse"); m.fit(X, t, epochs=25, batch_size=2, verbose=0)
        Xte = np.array([s[i - lookback:i] for i in range(lo, hi)]).reshape(-1, lookback, 1)
        runs.append(metrics(y[lo:hi], scaler.inverse_transform(m.predict(Xte, verbose=0)).ravel()))
    mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    std = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    return mean, std

val = statistical_models(n_train, n_train + n_val)
test = statistical_models(n_train + n_val, n)
val["LSTM (prototype config, ReLU)"], _ = lstm_eval(n_train, n_train + n_val, n_train, "relu")
test["LSTM (prototype config, ReLU)"], lstm_std = lstm_eval(n_train + n_val, n, n_train + n_val, "relu")
test["LSTM (tanh)"], _ = lstm_eval(n_train + n_val, n, n_train + n_val, "tanh")
save_json({"weeks": n, "split": [n_train, n_val, n - n_train - n_val], "validation": val, "test": test,
           "lstm_test_std_over_seeds": lstm_std}, "forecast_evaluation.json")
for name, m in test.items():
    if isinstance(m, dict):
        print(f"{name:32s} MAE {m['MAE']:.0f}  RMSE {m['RMSE']:.0f}  MAPE {m['MAPE']:.1f}")
