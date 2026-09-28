import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb  # This is the Module for predicting Consumer Survey Demand behaviour from the Survey Response
import tensorflow as tf
from keras.layers import Dense, LSTM  # This LSTM Module Predict the desired Supplier
from keras.models import Sequential

# ---------------------------------------------------------------------------
# 0. Shared constants (columns, dropdown levels, default weights)
# ---------------------------------------------------------------------------

LEVELS = ["Low", "Medium", "High"]

CONSUMER_COLUMNS = [
    "Purchase Frequency", "Price Sensitivity", "Promotion Response",
    "Satisfaction", "Repurchase Intention", "Previous Purchase", "Represents",
]

HISTORICAL_COLUMNS = ["Date", "Demand", "Price", "Seasonal Factor", "Promotion"]

SUPPLIER_COLUMNS = [
    "Supplier", "Unit Price", "Lead Time (d)", "Quality %", "Capacity",
    "Delay Rate %", "Defect Rate %", "Reliability %", "Transport Cost",
]

WEIGHT_KEYS = ["cost", "quality", "delivery", "reliability", "availability"]

WEIGHT_LABELS = {
    "cost": "Cost",
    "quality": "Quality",
    "delivery": "Delivery",
    "reliability": "Reliability",
    "availability": "Availability",
}


def default_weights() -> dict:
    """Equal weight across all five criteria (sums to 1.0)."""
    n = len(WEIGHT_KEYS)
    return {k: 1.0 / n for k in WEIGHT_KEYS}


def redistribute_weights(weights: dict, changed_key: str, new_val: float) -> dict:
    """User dragged `changed_key` to `new_val` (0-1). Clamp it, then rescale
    every other weight proportionally so the full set still sums to 1.0."""
    new_val = float(min(1.0, max(0.0, new_val)))
    others = [k for k in WEIGHT_KEYS if k != changed_key]
    remaining = 1.0 - new_val
    other_total = sum(weights.get(k, 0.0) for k in others)

    updated = dict(weights)
    updated[changed_key] = new_val

    if other_total <= 1e-9:
        # Nothing to scale from (e.g. all others were 0) — split evenly.
        share = remaining / len(others) if others else 0.0
        for k in others:
            updated[k] = share
    else:
        for k in others:
            updated[k] = weights.get(k, 0.0) / other_total * remaining

    return updated


# ---------------------------------------------------------------------------
# 0b. Sample data (used to pre-populate the console / "Load sample data")
# ---------------------------------------------------------------------------

def sample_consumer_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Purchase Frequency": "High", "Price Sensitivity": "Low", "Promotion Response": "Medium",
         "Satisfaction": "High", "Repurchase Intention": "High", "Previous Purchase": 120, "Represents": 400},
        {"Purchase Frequency": "Medium", "Price Sensitivity": "Medium", "Promotion Response": "High",
         "Satisfaction": "Medium", "Repurchase Intention": "Medium", "Previous Purchase": 80, "Represents": 250},
        {"Purchase Frequency": "Low", "Price Sensitivity": "High", "Promotion Response": "Low",
         "Satisfaction": "Low", "Repurchase Intention": "Low", "Previous Purchase": 30, "Represents": 100},
    ], columns=CONSUMER_COLUMNS)


def sample_historical_df() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=8, freq="7D")
    demand = [980, 1010, 1050, 1005, 1090, 1120, 1150, 1180]
    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Demand": demand,
        "Price": [19.99] * 8,
        "Seasonal Factor": [1.0, 1.0, 1.05, 0.95, 1.1, 1.1, 1.15, 1.15],
        "Promotion": [False, False, True, False, False, True, False, True],
    }, columns=HISTORICAL_COLUMNS)


def sample_supplier_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Supplier": "Northwind Co.", "Unit Price": 8.50, "Lead Time (d)": 12, "Quality %": 96,
         "Capacity": 1500, "Delay Rate %": 4, "Defect Rate %": 2, "Reliability %": 95, "Transport Cost": 0.35},
        {"Supplier": "Fabrikam Ltd.", "Unit Price": 7.90, "Lead Time (d)": 18, "Quality %": 91,
         "Capacity": 1200, "Delay Rate %": 9, "Defect Rate %": 4, "Reliability %": 88, "Transport Cost": 0.42},
        {"Supplier": "Contoso Supply", "Unit Price": 9.25, "Lead Time (d)": 8, "Quality %": 98,
         "Capacity": 900, "Delay Rate %": 2, "Defect Rate %": 1, "Reliability %": 97, "Transport Cost": 0.55},
    ], columns=SUPPLIER_COLUMNS)


# ---------------------------------------------------------------------------
# 0c. File upload helpers (read / map / template)
# ---------------------------------------------------------------------------

def read_any_table(uploaded_file) -> pd.DataFrame:
    """Read an uploaded .xlsx/.xls/.csv Streamlit UploadedFile into a DataFrame."""
    name = getattr(uploaded_file, "name", "") or ""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    if ext in ("xlsx", "xls"):
        return pd.read_excel(uploaded_file)
    # Fall back to sniffing: try Excel, then CSV.
    try:
        return pd.read_excel(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _map_to_columns(raw: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """Case/whitespace-insensitive column matching from an uploaded sheet
    onto our expected schema. Unmatched target columns are created empty."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    lookup = {_normalize(c): c for c in raw.columns}

    out = pd.DataFrame()
    for col in target_columns:
        src = lookup.get(_normalize(col))
        out[col] = raw[src] if src is not None else np.nan
    return out


def _clean_level(value, default="Medium"):
    if pd.isna(value):
        return default
    text = str(value).strip().title()
    return text if text in LEVELS else default


def parse_consumer_upload(raw: pd.DataFrame) -> pd.DataFrame:
    mapped = _map_to_columns(raw, CONSUMER_COLUMNS)
    mapped = mapped.dropna(how="all")
    for col in ["Purchase Frequency", "Price Sensitivity", "Promotion Response",
                "Satisfaction", "Repurchase Intention"]:
        mapped[col] = mapped[col].apply(_clean_level)
    mapped["Previous Purchase"] = pd.to_numeric(mapped["Previous Purchase"], errors="coerce").fillna(0.0)
    mapped["Represents"] = pd.to_numeric(mapped["Represents"], errors="coerce").fillna(1.0)
    return mapped.reset_index(drop=True)


def parse_historical_upload(raw: pd.DataFrame) -> pd.DataFrame:
    mapped = _map_to_columns(raw, HISTORICAL_COLUMNS)
    mapped["Date"] = pd.to_datetime(mapped["Date"], errors="coerce")
    mapped = mapped.dropna(subset=["Date"])
    mapped["Date"] = mapped["Date"].dt.strftime("%Y-%m-%d")
    mapped["Demand"] = pd.to_numeric(mapped["Demand"], errors="coerce").fillna(0.0)
    mapped["Price"] = pd.to_numeric(mapped["Price"], errors="coerce").fillna(0.0)
    mapped["Seasonal Factor"] = pd.to_numeric(mapped["Seasonal Factor"], errors="coerce").fillna(1.0)
    mapped["Promotion"] = mapped["Promotion"].apply(
        lambda v: bool(v) if not pd.isna(v) else False
    )
    return mapped.reset_index(drop=True)


def parse_supplier_upload(raw: pd.DataFrame) -> pd.DataFrame:
    mapped = _map_to_columns(raw, SUPPLIER_COLUMNS)
    mapped = mapped.dropna(subset=["Supplier"])
    numeric_cols = [c for c in SUPPLIER_COLUMNS if c != "Supplier"]
    for col in numeric_cols:
        mapped[col] = pd.to_numeric(mapped[col], errors="coerce").fillna(0.0)
    mapped["Supplier"] = mapped["Supplier"].astype(str).str.strip()
    return mapped.reset_index(drop=True)


def template_bytes(kind: str) -> bytes:
    """Build a blank .xlsx template (headers only) for the given data kind."""
    columns_by_kind = {
        "consumer": CONSUMER_COLUMNS,
        "historical": HISTORICAL_COLUMNS,
        "supplier": SUPPLIER_COLUMNS,
    }
    columns = columns_by_kind.get(kind, [])
    buffer = io.BytesIO()
    pd.DataFrame(columns=columns).to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 0d. Multi-product support: raw per-order supplier transactions
# ---------------------------------------------------------------------------
# Some supplier files aren't already aggregated one-row-per-supplier -- they're
# raw purchase-order history (one row per order) tagged with a Product_Id.
# These helpers detect that shape, list the products it covers, and derive a
# per-product SUPPLIER_COLUMNS table on demand using the exact same real-data
# derivation rules used for the single-product version of this dataset:
#   - Unit Price, Lead Time      -> real means from the order records
#   - Delay Rate                 -> % of orders whose lead time exceeds the
#                                    75th-percentile lead time (a standard
#                                    continuous-to-binary "late" threshold,
#                                    recomputed within the selected product's
#                                    own orders)
#   - Defect Rate, Reliability   -> real, from Defective_Units/Quantity and
#                                    the Compliance flag
#   - Quality                    -> 100 - Defect Rate (no quality-score field
#                                    exists in the source data)
#   - Capacity, Transport Cost   -> synthetic (no equivalent in the source
#                                    data); Capacity is anchored to that
#                                    supplier's real order volume for the
#                                    selected product rather than pure noise

_PRODUCT_ID_ALIASES = ["Product_Id", "Product ID", "ProductId", "Product", "SKU", "Item ID"]
_RAW_ORDER_MARKERS = ["Order_Date", "Delivery_Date", "Quantity", "Defective_Units"]


def _find_column(df: pd.DataFrame, aliases: list):
    lookup = {_normalize(c): c for c in df.columns}
    for name in aliases:
        hit = lookup.get(_normalize(name))
        if hit is not None:
            return hit
    return None


def is_raw_supplier_transactions(df: pd.DataFrame) -> bool:
    """True if this looks like one-row-per-order purchase history rather
    than an already-aggregated one-row-per-supplier table."""
    if df is None or df.empty:
        return False
    hits = sum(1 for marker in _RAW_ORDER_MARKERS if _find_column(df, [marker]) is not None)
    return hits >= 3


def get_product_column(df: pd.DataFrame):
    """Returns the actual Product-ID column name in df, or None if absent."""
    return _find_column(df, _PRODUCT_ID_ALIASES)


def list_products(df: pd.DataFrame) -> list:
    """Sorted list of distinct product IDs found in a raw transactions
    dataframe. Empty list if there's no product column."""
    col = get_product_column(df)
    if col is None:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def aggregate_historical_across_products(raw_df: pd.DataFrame) -> pd.DataFrame:
    """When 'All products' is selected and the historical upload covers
    several products, naively concatenating their rows would hand the trend
    fit several different Demand values for the same Date -- which corrupts
    the regression. Instead, sum Demand per Date across products (the
    correct way to represent total business-wide demand), and carry Price /
    Seasonal Factor as an average and Promotion as "any product on promo".
    """
    date_col = _find_column(raw_df, ["Date"])
    demand_col = _find_column(raw_df, ["Demand"])
    price_col = _find_column(raw_df, ["Price"])
    seasonal_col = _find_column(raw_df, ["Seasonal Factor"])
    promo_col = _find_column(raw_df, ["Promotion"])

    if date_col is None or demand_col is None:
        return raw_df

    df = raw_df.copy()
    df["_date_parsed"] = pd.to_datetime(df[date_col], errors="coerce")
    df[demand_col] = pd.to_numeric(df[demand_col], errors="coerce").fillna(0.0)

    agg = {demand_col: "sum"}
    if price_col:
        agg[price_col] = "mean"
    if seasonal_col:
        agg[seasonal_col] = "mean"
    if promo_col:
        agg[promo_col] = lambda s: s.astype(str).str.strip().str.lower().isin(
            ["true", "yes", "1"]).any()

    combined = df.groupby("_date_parsed").agg(agg).reset_index()
    combined = combined.rename(columns={"_date_parsed": date_col})
    combined[date_col] = combined[date_col].dt.strftime("%Y-%m-%d")
    return combined.sort_values(date_col).reset_index(drop=True)


def derive_supplier_table_from_raw(raw_df: pd.DataFrame, product_id: str = None,
                                   seed: int = 7) -> pd.DataFrame:
    """Aggregate raw per-order transactions into the standard SUPPLIER_COLUMNS
    shape, exactly the way the original real-data derivation was done --
    optionally restricted to a single product first.

    product_id=None aggregates across every product (legacy / all-products
    behaviour). A given product_id filters to that product's own orders
    before aggregating, so Lead Time, Delay Rate, Defect Rate and Reliability
    all reflect that product's real order history specifically.
    """
    rng = np.random.default_rng(seed)
    df = raw_df.copy()

    supplier_col = _find_column(df, ["Supplier", "Supplier info", "Supplier ID", "Supplier_Id"])
    order_date_col = _find_column(df, ["Order_Date", "Order Date"])
    delivery_date_col = _find_column(df, ["Delivery_Date", "Delivery Date"])
    quantity_col = _find_column(df, ["Quantity"])
    unit_price_col = _find_column(df, ["Unit_Price", "Unit Price"])
    defective_col = _find_column(df, ["Defective_Units", "Defective Units"])
    compliance_col = _find_column(df, ["Compliance"])
    product_col = get_product_column(df)

    if product_id is not None and product_col is not None:
        df = df[df[product_col].astype(str) == str(product_id)].copy()

    if df.empty or supplier_col is None:
        return pd.DataFrame(columns=SUPPLIER_COLUMNS)

    df[order_date_col] = pd.to_datetime(df[order_date_col], errors="coerce")
    df[delivery_date_col] = pd.to_datetime(df[delivery_date_col], errors="coerce")
    df["_lead_time_days"] = (df[delivery_date_col] - df[order_date_col]).dt.days

    valid_lead = df[df["_lead_time_days"].between(0, 60)]
    delay_threshold = valid_lead["_lead_time_days"].quantile(0.75) if len(valid_lead) else 15.0

    rows = []
    for supplier in sorted(df[supplier_col].dropna().unique()):
        s_all = df[df[supplier_col] == supplier]
        s_lead = valid_lead[valid_lead[supplier_col] == supplier]

        unit_price = s_all[unit_price_col].mean() if unit_price_col else 0.0
        lead_time = s_lead["_lead_time_days"].mean() if len(s_lead) else float(delay_threshold)
        delay_rate = (s_lead["_lead_time_days"] > delay_threshold).mean() * 100 if len(s_lead) else 0.0

        total_qty = s_all[quantity_col].sum() if quantity_col else 0.0
        total_defects = s_all[defective_col].sum() if defective_col else 0.0
        defect_rate = (total_defects / total_qty * 100) if total_qty else 0.0
        reliability = (s_all[compliance_col].astype(str).str.strip().str.lower() == "yes").mean() * 100 \
            if compliance_col else 0.0
        quality = max(60.0, 100 - defect_rate)

        capacity = int((total_qty / 90) * rng.uniform(0.8, 1.3)) if total_qty else 0
        transport_cost = round(rng.uniform(120, 420), 2)

        rows.append({
            "Supplier": str(supplier),
            "Unit Price": round(float(unit_price), 2),
            "Lead Time (d)": round(float(lead_time), 1),
            "Quality %": round(float(quality), 1),
            "Capacity": capacity,
            "Delay Rate %": round(float(delay_rate), 1),
            "Defect Rate %": round(float(defect_rate), 1),
            "Reliability %": round(float(reliability), 1),
            "Transport Cost": transport_cost,
        })

    return pd.DataFrame(rows, columns=SUPPLIER_COLUMNS)


# ---------------------------------------------------------------------------
# 1. Real XGBoost Model for Consumer Demand
# ---------------------------------------------------------------------------

def compute_consumer_demand(df: pd.DataFrame) -> float:
    """Proxy for the XGBoost consumer-behavior model: converts each row's
    categorical features into a demand multiplier, applies it to that
    consumer's previous purchase quantity, and scales by segment size."""
    if df is None or df.empty:
        return 0.0

    work = df.copy()
    # Feature Engineering/ Dummy variable encoding for categorical Survey fields
    categorical_cols = ["Purchase Frequency", "Price Sensitivity", "Promotion Response", "Satisfaction",
                        "Repurchase Intention"]
    # Create training/inference feature matrix X
    # For a cold-start prototype without a pre-trained .json model file,
    # we train an instant XGBRegressor on the fly using synthetic target signals
    # derived from the features, or load a pre-trained model if available
    # Guard against missing categorical columns (e.g. after a bad upload)
    for col in categorical_cols:
        if col not in work.columns:
            work[col] = "Medium"
    encoded_df = pd.get_dummies(work[categorical_cols], drop_first=True)
    encoded_df["Previous Purchase"] = work["Previous Purchase"].astype(float)
    encoded_df["Segment Size"] = work["Represents"].astype(float)
    # Generate a baseline proxy target for the model to train on if no pre-trained weights exist yet
    # (In production, replace this block with: model = xgb.Booster(); model.load_model("xgb_consumer_model.json"))
    y_synthetic = (encoded_df["Previous Purchase"] * encoded_df.get("Segment Size", 50) * 1.1)

    # Train the model using synthetic target signals
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(encoded_df, y_synthetic)

    # Use the trained model to predict consumer demand
    y_pred = model.predict(encoded_df)
    total_pred = np.sum(y_pred * (work["Represents"].astype(float) / max(1, work["Represents"].astype(float).mean())))
    return float(max(0.0, total_pred))


# ---------------------------------------------------------------------------
# 2. Real LSTM Neural Network Model for Time-Series Historical Forecast
# ---------------------------------------------------------------------------

def compute_historical_forecast(df: pd.DataFrame, horizon_days: float):
    """ Proxy for the LSTM model: fits a straight line through the demand
    history and projects it forward by the forecast horizon."""
    if df is None or df.empty:
        return 0.0, False, None

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date", "Demand"]).sort_values("Date").reset_index(drop=True)
    n = len(work)

    if n == 0:
        return 0.0, False, None
    if n < 4:
        # Fallback to linear fit if not enough points to form LSTM sequences
        x = np.arange(n)
        y = work["Demand"].astype(float).values
        slope, intercept = np.polyfit(x, y, 1)
        return max(0.0, float(intercept + slope * (n - 1 + (horizon_days / 7)))), True, 7.0

    # Prepare data for LSTM (Normalization)
    demand_values = work["Demand"].astype(float).values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(demand_values)

    # Create sliding window sequences (lookback = 3 steps)
    lookback = min(3, n - 1)
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build a lightweight LSTM network
    tf.keras.utils.set_random_seed(42)
    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')

    # Train for a few epochs on the historical sequence
    lstm_model.fit(X, y, epochs=25, batch_size=2, verbose=0)

    # Recursively forecast forward across the horizon
    current_window = scaled_data[-lookback:].reshape(1, lookback, 1)
    steps_ahead = max(1, int(horizon_days / 7))

    future_scaled_predictions = []
    for _ in range(steps_ahead):
        pred = lstm_model.predict(current_window, verbose=0)[0][0]
        future_scaled_predictions.append(pred)
        # Slide window forward
        new_window = np.append(current_window[0, 1:, 0], pred)
        current_window = new_window.reshape(1, lookback, 1)

    # Inverse scale predictions back to original demand numbers
    unscaled_preds = scaler.inverse_transform(np.array(future_scaled_predictions).reshape(-1, 1))
    final_forecast = float(unscaled_preds[-1][0])

    steps = work["Date"].diff().dt.days.dropna()
    avg_step = float(steps.mean()) if len(steps) else 7.0

    return max(0.0, final_forecast), True, avg_step


# ---------------------------------------------------------------------------
# 3. Hybrid forecast blend
# ---------------------------------------------------------------------------

"""From the Consumer Behaviour Function and Historical Forecast Function we get a demand, That Demand We Put into a
formula   (DHybrid = alpha * Dc + (1-alpha) Dh

From there We Get a output of Overall actual Demand"""


def compute_hybrid_demand(consumer_demand: float, historical_forecast: float,
                          alpha: float, scenario: float) -> float:
    """D_hybrid = (alpha * D_consumer + (1 - alpha) * D_historical) * scenario."""
    alpha = float(min(1.0, max(0.0, alpha)))
    blended = alpha * float(consumer_demand) + (1 - alpha) * float(historical_forecast)
    return max(0.0, blended * float(scenario))


# ---------------------------------------------------------------------------
# 4. Supplier feasibility & multi-criteria ranking
# ---------------------------------------------------------------------------
""" Here I Check the First Criteria of If the Supplier is Capable of the work. If That Supplier fulfills the Demand Capacity then he is In on the Competition.
On the other Hand, If the Supplier, Doesn't Fulfills the Demand Capacity then he is Out."""


def apply_feasibility(supplier_df: pd.DataFrame, hybrid_demand: float) -> pd.DataFrame:
    """A supplier is feasible if it can cover the forecasted demand."""
    out = supplier_df.copy()
    out["Capacity"] = pd.to_numeric(out["Capacity"], errors="coerce").fillna(0.0)
    out["Feasible"] = out["Capacity"] >= float(hybrid_demand)
    return out


def _normalize_series(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """Min-max normalize a metric to 0-1. Falls back to a neutral 1.0 for
    every row when all values are equal (nothing to discriminate on)."""
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(1.0, index=s.index)
    scaled = (s - lo) / (hi - lo)
    return scaled if higher_is_better else 1 - scaled


""" Here I Rank the Supplier with the Scores with their proposed Price, Lead Time, Quality %, Delay Rate %, Defect Rate %,
Reliability %, Capacity, Transport Cost

From these Information I Calculate their Score and Rank Them"""


def compute_ranking(supplier_df: pd.DataFrame, hybrid_demand: float, weights: dict):
    """Returns (full_result_df, ranked_df). full_result_df carries every
    supplier plus feasibility + score; ranked_df is feasible suppliers only,
    sorted by Score descending with a fresh index."""
    if supplier_df is None or supplier_df.empty:
        empty = supplier_df.copy() if supplier_df is not None else pd.DataFrame(columns=SUPPLIER_COLUMNS)
        return empty, empty

    full = apply_feasibility(supplier_df, hybrid_demand)

    for col in ["Unit Price", "Lead Time (d)", "Quality %", "Delay Rate %",
                "Defect Rate %", "Reliability %", "Capacity", "Transport Cost"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)

    cost_score = _normalize_series(full["Unit Price"] + full["Transport Cost"], higher_is_better=False)
    quality_score = _normalize_series(full["Quality %"], higher_is_better=True)
    delivery_raw = full["Lead Time (d)"].rank(pct=True) + full["Delay Rate %"].rank(pct=True)
    delivery_score = _normalize_series(delivery_raw, higher_is_better=False)
    reliability_adj = full["Reliability %"] - full["Defect Rate %"]
    reliability_score = _normalize_series(reliability_adj, higher_is_better=True)
    availability_raw = full["Capacity"] - float(hybrid_demand)
    availability_score = _normalize_series(availability_raw, higher_is_better=True)

    full["Score"] = (
            weights.get("cost", 0.0) * cost_score
            + weights.get("quality", 0.0) * quality_score
            + weights.get("delivery", 0.0) * delivery_score
            + weights.get("reliability", 0.0) * reliability_score
            + weights.get("availability", 0.0) * availability_score
    )

    ranked = (
        full[full["Feasible"]]
        .sort_values("Score", ascending=False)
        .reset_index(drop=True)
    )
    return full, ranked


""" This Function give me the Reason why I Select that Supplier for that Demand"""


def generate_reasons(top: pd.Series, ranked: pd.DataFrame) -> list:
    """Short human-readable bullets explaining why `top` was recommended."""
    reasons = []

    if len(ranked) > 1:
        if top["Unit Price"] <= ranked["Unit Price"].min() + 1e-9:
            reasons.append(f"Lowest unit price among feasible suppliers (${top['Unit Price']:.2f}).")
        if top["Quality %"] >= ranked["Quality %"].max() - 1e-9:
            reasons.append(f"Highest reported quality rate ({top['Quality %']:.0f}%).")
        if top["Lead Time (d)"] <= ranked["Lead Time (d)"].min() + 1e-9:
            reasons.append(f"Shortest lead time ({top['Lead Time (d)']:.0f} days).")
        if top["Capacity"] >= ranked["Capacity"].max() - 1e-9:
            reasons.append(f"Most spare capacity above the forecast ({top['Capacity']:.0f} units).")
        if (top["Reliability %"] - top["Defect Rate %"]) >= (
                ranked["Reliability %"] - ranked["Defect Rate %"]).max() - 1e-9:
            reasons.append(
                f"Best reliability once defects are factored in ({top['Reliability %']:.0f}% reliability, {top['Defect Rate %']:.0f}% defects).")

    if not reasons:
        reasons.append(
            f"Best overall weighted score ({top['Score'] * 100:.1f}) across cost, quality, delivery, reliability and availability.")

    return reasons[:4]