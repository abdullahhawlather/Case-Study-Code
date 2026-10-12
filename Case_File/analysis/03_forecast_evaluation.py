"""Chronological multi-step evaluation of weekly order-quantity forecasts.

Models: naive last week, 4-week moving average, training mean, simple exponential
smoothing (SES), Holt damped-trend exponential smoothing, ARIMA(1,0,1), and an
LSTM neural baseline using the prototype configuration (32 units, ReLU, Dense(1),
Adam/MSE, 3-week lookback, min-max scaling, batch size 2).

The first 70% of weeks train the models, the next 15% form validation, and the final
15% form the test set. Each split is forecast from a fixed origin without using
actual values from inside its target horizon. LSTM scores are averaged across seeds.

Usage: python 03_forecast_evaluation.py [--seeds 5] [--epochs 25]
Output: results/forecast_evaluation.json
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from common import load_orders, save_json

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds for LSTM")
parser.add_argument("--epochs", type=int, default=25, help="Training epochs per LSTM seed")
args = parser.parse_args()

orders = load_orders()
orders["Order_Date"] = pd.to_datetime(orders["Order_Date"], errors="coerce")
orders["Quantity"] = pd.to_numeric(orders["Quantity"], errors="coerce")
orders = orders.dropna(subset=["Order_Date", "Quantity"])
weekly = orders.set_index("Order_Date")["Quantity"].resample("W-MON").sum().sort_index()
weekly = weekly.loc[weekly.index < pd.Timestamp("2024-01-02")]
y = weekly.to_numpy(dtype=float)
n = len(y)
if n < 12:
    raise ValueError(f"Need at least 12 weekly observations for chronological evaluation; found {n}.")
n_train, n_val = int(n * .70), int(n * .15)
n_test = n - n_train - n_val
print(f"weeks: {n}; split: train={n_train}, validation={n_val}, test={n_test}")


def metrics(actual, forecast):
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    denom = np.maximum(np.abs(a), 1.0)
    smape_denom = np.maximum(np.abs(a) + np.abs(f), 1e-9)
    return {
        "MAE": float(np.mean(np.abs(a - f))),
        "RMSE": float(np.sqrt(np.mean((a - f) ** 2))),
        "MAPE": float(np.mean(np.abs(a - f) / denom) * 100),
        "sMAPE": float(np.mean(2 * np.abs(a - f) / smape_denom) * 100),
        "WAPE": float(np.sum(np.abs(a - f)) / max(np.sum(np.abs(a)), 1e-9) * 100),
    }


def forecast_models(train, horizon):
    train = np.asarray(train, dtype=float)
    horizon = int(horizon)
    preds = {
        "Naive (last week)": np.full(horizon, train[-1], dtype=float),
        "Moving average (4 wk)": np.full(horizon, np.mean(train[-min(4, len(train)):]), dtype=float),
        "Training mean": np.full(horizon, np.mean(train), dtype=float),
    }
    try:
        preds["Exp. smoothing (SES)"] = np.asarray(
            SimpleExpSmoothing(train, initialization_method="estimated").fit(optimized=True).forecast(horizon), dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        preds["Exp. smoothing (SES)"] = np.full(horizon, train[-1], dtype=float)
    try:
        if len(train) >= 3 and np.ptp(train) >= 1e-12:
            preds["Holt damped trend"] = np.asarray(
                Holt(train, damped_trend=True, initialization_method="estimated").fit(optimized=True).forecast(horizon), dtype=float)
        else:
            preds["Holt damped trend"] = np.full(horizon, train[-1], dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        preds["Holt damped trend"] = np.full(horizon, train[-1], dtype=float)
    try:
        if len(train) >= 8:
            preds["ARIMA(1,0,1)"] = np.asarray(ARIMA(train, order=(1, 0, 1)).fit().forecast(horizon), dtype=float)
        else:
            preds["ARIMA(1,0,1)"] = np.full(horizon, train[-1], dtype=float)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        preds["ARIMA(1,0,1)"] = np.full(horizon, train[-1], dtype=float)
    return {name: np.clip(np.nan_to_num(values, nan=float(train[-1]), posinf=float(train[-1]), neginf=0.0), 0.0, None)
            for name, values in preds.items()}


def lstm_forecast(train, horizon, seed, epochs):
    """Recursive LSTM forecast trained only on observations before the split origin."""
    # Import lazily so non-neural benchmarks remain usable if TensorFlow is unavailable.
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense
    except ImportError as exc:
        raise ImportError("LSTM comparison requires TensorFlow. Install it with `pip install tensorflow`.") from exc

    lookback = 3
    train = np.asarray(train, dtype=float)
    if len(train) <= lookback:
        raise ValueError(f"LSTM requires more than {lookback} training observations.")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train.reshape(-1, 1)).ravel()
    X = np.array([scaled[i - lookback:i] for i in range(lookback, len(scaled))], dtype=float).reshape(-1, lookback, 1)
    target = scaled[lookback:]
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = Sequential([LSTM(32, activation="relu", input_shape=(lookback, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, target, epochs=epochs, batch_size=2, verbose=0, shuffle=False)

    window = scaled[-lookback:].copy()
    forecast_scaled = []
    for _ in range(int(horizon)):
        x = window.reshape(1, lookback, 1)
        pred = float(model.predict(x, verbose=0)[0, 0])
        forecast_scaled.append(pred)
        window = np.append(window[1:], pred)
    forecast = scaler.inverse_transform(np.asarray(forecast_scaled).reshape(-1, 1)).ravel()
    return np.clip(np.nan_to_num(forecast, nan=float(train[-1]), posinf=float(train[-1]), neginf=0.0), 0.0, None)


def evaluate_split(train_end, target_start, target_end):
    actual = y[target_start:target_end]
    train = y[:train_end]
    predictions = forecast_models(train, len(actual))
    scores = {name: metrics(actual, forecast) for name, forecast in predictions.items()}
    lstm_runs = []
    for seed in range(max(1, args.seeds)):
        forecast = lstm_forecast(train, len(actual), seed, args.epochs)
        lstm_runs.append(metrics(actual, forecast))
    # Keep the mean metric in the main comparison and report seed variability separately.
    scores["LSTM (32 units, ReLU; mean over seeds)"] = {
        metric: float(np.mean([run[metric] for run in lstm_runs]))
        for metric in lstm_runs[0]
    }
    seed_std = {metric: float(np.std([run[metric] for run in lstm_runs])) for metric in lstm_runs[0]}
    return scores, seed_std


validation, validation_lstm_std = evaluate_split(n_train, n_train, n_train + n_val)
test, test_lstm_std = evaluate_split(n_train + n_val, n_train + n_val, n)
best = min(test, key=lambda name: test[name]["WAPE"])
payload = {
    "target": "weekly total order quantity",
    "historical_model": "Holt damped-trend exponential smoothing",
    "evaluation_design": "chronological fixed-origin multi-step forecast; no target-horizon observations used for fitting",
    "lstm_configuration": {"units": 32, "activation": "relu", "lookback_weeks": 3, "epochs": args.epochs,
                           "batch_size": 2, "optimizer": "adam", "loss": "mse", "scaling": "MinMaxScaler",
                           "seeds": max(1, args.seeds), "forecasting": "recursive multi-step"},
    "weeks": n,
    "date_start": str(weekly.index.min().date()),
    "date_end": str(weekly.index.max().date()),
    "split": {"train": n_train, "validation": n_val, "test": n_test},
    "validation": validation,
    "test": test,
    "lstm_seed_std": {"validation": validation_lstm_std, "test": test_lstm_std},
    "best_test_model_by_WAPE": best,
}
save_json(payload, "forecast_evaluation.json")
for name, score in test.items():
    print(f"{name:42s} MAE {score['MAE']:.2f} RMSE {score['RMSE']:.2f} "
          f"sMAPE {score['sMAPE']:.2f}% WAPE {score['WAPE']:.2f}%")
print("Best test model by WAPE:", best)
