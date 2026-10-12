import math
import io
import numpy as np
import pandas as pd
import xgboost as xgb  # This is the Module for predicting Consumer Survey Demand behaviour from the Survey Response

# ---------------------------------------------------------------------------
# 0. Shared constants (columns, dropdown levels, default weights)
# ---------------------------------------------------------------------------

LEVELS = ["Low", "Medium", "High"]

CONSUMER_COLUMNS = [
    "Purchase Frequency", "Price Sensitivity", "Promotion Response",
    "Satisfaction", "Repurchase Intention", "Previous Purchase", "Represents",
    "Expected Volume Change",
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
         "Satisfaction": "High", "Repurchase Intention": "High", "Previous Purchase": 12, "Represents": 4, "Expected Volume Change": "Significant increase (>20% growth)"},
        {"Purchase Frequency": "Medium", "Price Sensitivity": "Medium", "Promotion Response": "High",
         "Satisfaction": "Medium", "Repurchase Intention": "Medium", "Previous Purchase": 8, "Represents": 3, "Expected Volume Change": "Moderate increase (1–20% growth)"},
        {"Purchase Frequency": "Low", "Price Sensitivity": "High", "Promotion Response": "Low",
         "Satisfaction": "Low", "Repurchase Intention": "Low", "Previous Purchase": 3, "Represents": 2, "Expected Volume Change": "Decrease"},
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
         "Capacity": 6000, "Delay Rate %": 4, "Defect Rate %": 2, "Reliability %": 95, "Transport Cost": 0.35},
        {"Supplier": "Fabrikam Ltd.", "Unit Price": 7.90, "Lead Time (d)": 18, "Quality %": 91,
         "Capacity": 5000, "Delay Rate %": 9, "Defect Rate %": 4, "Reliability %": 88, "Transport Cost": 0.42},
        {"Supplier": "Contoso Supply", "Unit Price": 9.25, "Lead Time (d)": 8, "Quality %": 98,
         "Capacity": 4500, "Delay Rate %": 2, "Defect Rate %": 1, "Reliability %": 97, "Transport Cost": 0.55},
    ], columns=SUPPLIER_COLUMNS)


# ---------------------------------------------------------------------------
# 0c. File upload helpers (read / map / template)
# ---------------------------------------------------------------------------

def read_any_table(uploaded_file) -> pd.DataFrame:
    """Read an uploaded .xlsx/.xls/.csv Streamlit UploadedFile into a DataFrame."""
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)  # Streamlit reruns re-read the same buffer
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


COLUMN_ALIASES = {
    "Supplier": ["Supplier Name", "Supplier_Name", "Supplier info", "Supplier ID", "Supplier_Id"],
    "Lead Time (d)": ["Lead Time", "Average Lead Time", "Avg Lead Time", "Lead Time Days"],
    "Quality %": ["Quality", "Quality Score", "Quality_Score"],
    "Delay Rate %": ["Historical Delay Rate", "Delay Rate", "Delay Rate (%)"],
    "Defect Rate %": ["Defect Rate", "Avg Defect Rate", "Avg_Defect_Rate"],
    "Reliability %": ["Reliability", "Compliance Rate", "Compliance_Rate"],
    "Transport Cost": ["Transportation Cost", "Transport Cost Per Unit"],
    "Expected Volume Change": ["Expected Purchase Behavior", "Expected Volume Change", "Volume Change"],
    "Product ID": ["Product_ID", "Product_Id", "ProductId", "SKU", "Item ID"],
}


def _map_to_columns(raw: pd.DataFrame, target_columns: list) -> pd.DataFrame:
    """Map common header aliases to the expected schema, ignoring case/punctuation.
    Missing optional columns are left empty so parsers can apply explicit defaults."""
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    lookup = {_normalize(c): c for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for col in target_columns:
        candidates = [col] + COLUMN_ALIASES.get(col, [])
        src = next((lookup[_normalize(name)] for name in candidates if _normalize(name) in lookup), None)
        out[col] = raw[src] if src is not None else np.nan
    return out


def _clean_level(value, default="Medium"):
    if pd.isna(value):
        return default
    text = str(value).strip().title()
    return text if text in LEVELS else default


def _to_bool(value) -> bool:
    """Robust truthiness for uploaded cells. bool("False") is True in Python,
    so strings must be interpreted explicitly."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1", "t")
    return bool(value)


def parse_consumer_upload(raw: pd.DataFrame) -> pd.DataFrame:
    mapped = _map_to_columns(raw, CONSUMER_COLUMNS)
    mapped = mapped.dropna(how="all")
    for col in ["Purchase Frequency", "Price Sensitivity", "Promotion Response",
                "Satisfaction", "Repurchase Intention"]:
        mapped[col] = mapped[col].apply(_clean_level)
    mapped["Previous Purchase"] = pd.to_numeric(mapped["Previous Purchase"], errors="coerce").fillna(0.0)
    mapped["Represents"] = pd.to_numeric(mapped["Represents"], errors="coerce").fillna(1.0).clip(lower=0.0)
    valid_volume_changes = set(VOLUME_CHANGE_MIDPOINTS)
    if "Expected Volume Change" not in mapped:
        mapped["Expected Volume Change"] = "Remain stable / unchanged"
    mapped["Expected Volume Change"] = mapped["Expected Volume Change"].where(
        mapped["Expected Volume Change"].isin(valid_volume_changes), "Remain stable / unchanged"
    )
    return mapped[CONSUMER_COLUMNS].reset_index(drop=True)


def parse_historical_upload(raw: pd.DataFrame) -> pd.DataFrame:
    mapped = _map_to_columns(raw, HISTORICAL_COLUMNS)
    mapped["Date"] = pd.to_datetime(mapped["Date"], errors="coerce")
    mapped = mapped.dropna(subset=["Date"])
    mapped["Date"] = mapped["Date"].dt.strftime("%Y-%m-%d")
    mapped["Demand"] = pd.to_numeric(mapped["Demand"], errors="coerce").fillna(0.0)
    mapped["Price"] = pd.to_numeric(mapped["Price"], errors="coerce").fillna(0.0)
    mapped["Seasonal Factor"] = pd.to_numeric(mapped["Seasonal Factor"], errors="coerce").fillna(1.0)
    mapped["Promotion"] = mapped["Promotion"].apply(_to_bool)
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
    """Aggregate purchase-order records into supplier metrics.

    Observed fields are calculated from order history. Capacity is explicitly a
    proxy (largest non-cancelled historical order quantity) because the source
    has no supplier capacity field. Transport cost is an estimate (5% of mean
    negotiated/unit price) because transport charges are absent from the source.
    """
    df = raw_df.copy()
    supplier_col = _find_column(df, ["Supplier", "Supplier info", "Supplier ID", "Supplier_Id", "Supplier_Name"])
    order_date_col = _find_column(df, ["Order_Date", "Order Date"])
    delivery_date_col = _find_column(df, ["Delivery_Date", "Delivery Date"])
    quantity_col = _find_column(df, ["Quantity"])
    unit_price_col = _find_column(df, ["Negotiated_Price", "Negotiated Price", "Unit_Price", "Unit Price"])
    fallback_price_col = _find_column(df, ["Unit_Price", "Unit Price"])
    defective_col = _find_column(df, ["Defective_Units", "Defective Units"])
    compliance_col = _find_column(df, ["Compliance"])
    status_col = _find_column(df, ["Delivery_Status", "Delivery Status", "Order_Status", "Order Status"])
    product_col = get_product_column(df)

    if product_id is not None and product_col is not None:
        df = df[df[product_col].astype(str).str.strip() == str(product_id).strip()].copy()
    if df.empty or supplier_col is None:
        return pd.DataFrame(columns=SUPPLIER_COLUMNS)

    if order_date_col and delivery_date_col:
        df[order_date_col] = pd.to_datetime(df[order_date_col], errors="coerce")
        df[delivery_date_col] = pd.to_datetime(df[delivery_date_col], errors="coerce")
        df["_lead_time_days"] = (df[delivery_date_col] - df[order_date_col]).dt.days
        valid_lead = df[df["_lead_time_days"].between(0, 60)].copy()
    else:
        df["_lead_time_days"] = np.nan
        valid_lead = df.iloc[0:0].copy()

    if status_col:
        delivered_mask = df[status_col].astype(str).str.strip().str.lower().isin(
            ["delivered", "partially delivered", "partial", "complete", "completed"]
        )
        usable = df[delivered_mask].copy()
        if usable.empty:
            usable = df.copy()
    else:
        usable = df.copy()

    # A relative late threshold is used only when the source does not provide a
    # trustworthy explicit late/on-time flag. It is a proxy, not an official SLA.
    threshold = valid_lead["_lead_time_days"].quantile(0.75) if len(valid_lead) else 15.0
    rows = []
    for supplier in sorted(df[supplier_col].dropna().astype(str).unique()):
        s_all = df[df[supplier_col].astype(str) == supplier]
        s_use = usable[usable[supplier_col].astype(str) == supplier]
        s_lead = valid_lead[valid_lead[supplier_col].astype(str) == supplier]

        price_source = unit_price_col if unit_price_col in s_use.columns else fallback_price_col
        prices = pd.to_numeric(s_use[price_source], errors="coerce") if price_source else pd.Series(dtype=float)
        unit_price = float(prices.dropna().mean()) if prices.notna().any() else 0.0
        lead_time = float(s_lead["_lead_time_days"].mean()) if len(s_lead) else 0.0
        delay_rate = float((s_lead["_lead_time_days"] > threshold).mean() * 100) if len(s_lead) else 0.0

        qty = pd.to_numeric(s_use[quantity_col], errors="coerce") if quantity_col else pd.Series(dtype=float)
        total_qty = float(qty.clip(lower=0).sum()) if qty.notna().any() else 0.0
        defects = pd.to_numeric(s_use[defective_col], errors="coerce") if defective_col else pd.Series(dtype=float)
        total_defects = float(defects.clip(lower=0).sum()) if defects.notna().any() else 0.0
        defect_rate = min(100.0, total_defects / total_qty * 100) if total_qty > 0 else 0.0
        if compliance_col and len(s_use):
            reliability = float(s_use[compliance_col].astype(str).str.strip().str.lower().isin(["yes", "true", "1", "compliant"]).mean() * 100)
        else:
            reliability = 0.0
        quality = max(0.0, 100.0 - defect_rate)
        # Capacity is not present in the raw source. Estimate a transparent
        # 30-day supply proxy from the maximum rolling 30-day quantity in the
        # supplier's non-cancelled order history; do not fabricate random values.
        capacity = int(qty.max()) if qty.notna().any() else 0
        if quantity_col and order_date_col and len(s_use):
            cap_rows = s_use[[order_date_col, quantity_col]].copy()
            cap_rows[order_date_col] = pd.to_datetime(cap_rows[order_date_col], errors="coerce")
            cap_rows[quantity_col] = pd.to_numeric(cap_rows[quantity_col], errors="coerce").fillna(0).clip(lower=0)
            cap_rows = cap_rows.dropna(subset=[order_date_col]).sort_values(order_date_col)
            if len(cap_rows):
                daily = cap_rows.groupby(order_date_col)[quantity_col].sum().sort_index()
                rolling = daily.rolling("30D").sum()
                if len(rolling) and np.isfinite(rolling.max()):
                    capacity = max(capacity, int(rolling.max()))
        transport_cost = round(unit_price * 0.05, 2)  # explicitly estimated proxy

        rows.append({
            "Supplier": supplier,
            "Unit Price": round(unit_price, 2),
            "Lead Time (d)": round(lead_time, 1),
            "Quality %": round(quality, 1),
            "Capacity": capacity,
            "Delay Rate %": round(delay_rate, 1),
            "Defect Rate %": round(defect_rate, 2),
            "Reliability %": round(reliability, 1),
            "Transport Cost": transport_cost,
        })
    return pd.DataFrame(rows, columns=SUPPLIER_COLUMNS)


# ---------------------------------------------------------------------------
# 1. Real XGBoost Model for Consumer Demand
# ---------------------------------------------------------------------------

LEVEL_SCORE = {"Low": 0, "Medium": 1, "High": 2}


def compute_consumer_demand(df: pd.DataFrame, horizon_days: float = 30.0) -> float:
    """XGBoost consumer-behaviour model.

    Each respondent's expected quantity = Previous Purchase x a behaviour
    multiplier derived from the survey levels. An XGBRegressor learns that
    mapping from the (ordinal-encoded) answers, and the per-respondent
    predictions are then scaled up by how many consumers each row represents.
    """
    if df is None or df.empty:
        return 0.0

    work = df.copy()
    categorical_cols = ["Purchase Frequency", "Price Sensitivity", "Promotion Response",
                        "Satisfaction", "Repurchase Intention"]
    for col in categorical_cols:
        if col not in work.columns:
            work[col] = "Medium"

    # Ordinal encoding (Low < Medium < High). get_dummies(drop_first=True) threw
    # away the ordering and produced different columns for different uploads.
    X = pd.DataFrame(index=work.index)
    for col in categorical_cols:
        X[col] = work[col].map(LEVEL_SCORE).fillna(1).astype(float)
    prev = pd.to_numeric(work["Previous Purchase"], errors="coerce").fillna(0.0)
    X["Previous Purchase"] = prev
    # Approximate purchases within the selected horizon from the survey's
    # frequency bands: weekly≈4.345/month, monthly=1/month, low=quarterly proxy.
    frequency_multiplier = work["Purchase Frequency"].map({"High": 4.345, "Medium": 1.0, "Low": 1.0 / 3.0}).fillna(1.0)
    horizon_multiplier = max(0.0, float(horizon_days)) / 30.0
    volume_col = work.get("Expected Volume Change", pd.Series("Remain stable / unchanged", index=work.index))
    X["Expected Volume Change"] = volume_col.map(VOLUME_CHANGE_MIDPOINTS).fillna(0.0).astype(float)
    represents = pd.to_numeric(work["Represents"], errors="coerce").fillna(1.0).clip(lower=0.0)

    # Cold-start target (replace with a pre-trained model when real labels exist).
    # Higher frequency / promo response / satisfaction / repurchase raise demand,
    # higher price sensitivity lowers it. Segment size is NOT part of the
    # per-respondent target -- it is applied once, at the end.
    signal = (
        (X["Purchase Frequency"] - 1)
        - (X["Price Sensitivity"] - 1)
        + (X["Promotion Response"] - 1)
        + (X["Satisfaction"] - 1)
        + (X["Repurchase Intention"] - 1)
    )
    multiplier = (1.0 + 0.05 * signal + X["Expected Volume Change"]).clip(lower=0.5)
    y_per_respondent = prev * multiplier * frequency_multiplier * horizon_multiplier

    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                             min_child_weight=0, random_state=42)
    model.fit(X, y_per_respondent)
    y_pred = np.clip(model.predict(X), 0.0, None)

    # Each row stands for `Represents` consumers. (The old code multiplied by
    # Represents inside the target AND again here, double-counting segment size.)
    return float(max(0.0, np.sum(y_pred * represents)))


# ---------------------------------------------------------------------------
# 2. Forecast Historical demand with Holt's damped-trend exponential smoothing Method
# ---------------------------------------------------------------------------

def compute_historical_forecast(df: pd.DataFrame, horizon_days: float):
    """Forecast demand with Holt's damped-trend exponential smoothing.

    Returns (total horizon demand, used_history, avg_step_days), preserving App.py's interface.
    Forecasts each future model step within the selected horizon and sums them,
    because supplier capacity is compared with total demand over that horizon.
    Observations are treated as equally spaced model steps; provide regularly
    spaced demand observations where possible.
    """
    from statsmodels.tsa.holtwinters import Holt

    if df is None or df.empty:
        return 0.0, False, None

    work = df.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Demand"] = pd.to_numeric(work["Demand"], errors="coerce")
    work = work.dropna(subset=["Date", "Demand"]).sort_values("Date")
    work = work.groupby("Date", as_index=False)["Demand"].sum()
    y = work["Demand"].to_numpy(dtype=float)
    n = len(y)

    if n == 0:
        return 0.0, False, None

    date_steps = work["Date"].diff().dt.total_seconds().dropna() / 86400.0
    avg_step = float(date_steps.median()) if len(date_steps) else 7.0
    if not np.isfinite(avg_step) or avg_step <= 0:
        avg_step = 7.0
    steps_ahead = max(1, int(round(max(0.0, float(horizon_days)) / avg_step)))

    if n == 1:
        steps_ahead = max(1, int(round(max(0.0, float(horizon_days)) / avg_step)))
        return max(0.0, float(y[-1]) * steps_ahead), True, avg_step

    if n < 6 or np.ptp(y) < 1e-9:
        if n < 3:
            future = np.repeat(float(y[-1]), steps_ahead)
        else:
            slope = float(np.polyfit(np.arange(n, dtype=float), y, 1)[0])
            cap = max(float(np.mean(np.abs(y))) * 0.10, 1.0)
            slope = float(np.clip(slope, -cap, cap))
            future = np.array([max(0.0, float(y[-1] + slope * (i + 1) * 0.8)) for i in range(steps_ahead)])
        return max(0.0, float(future.sum())), True, avg_step

    try:
        model = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
        future = np.asarray(model.forecast(steps_ahead), dtype=float)
        forecast = float(np.clip(future, 0.0, None).sum())
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        slope = float(np.polyfit(np.arange(n, dtype=float), y, 1)[0])
        cap = max(float(np.mean(np.abs(y))) * 0.10, 1.0)
        slope = float(np.clip(slope, -cap, cap))
        future = np.array([max(0.0, float(y[-1] + slope * (i + 1) * 0.8)) for i in range(steps_ahead)])
        forecast = float(future.sum())

    if not np.isfinite(forecast):
        forecast = float(max(0.0, y[-1]) * steps_ahead)
    return max(0.0, forecast), True, avg_step


# ---------------------------------------------------------------------------
# 3. Hybrid forecast blend
# ---------------------------------------------------------------------------

"""
Implements the revised hybrid of the paper (Section 3.3):
    D_hybrid = D_h * (1 + alpha * g_c) * scenario
where g_c is the survey-implied relative change in demand.
From there We Get a output of Overall actual Demand"""

VOLUME_CHANGE_MIDPOINTS = {
    "Significant increase (>20% growth)": 0.25,
    "Moderate increase (1\u201320% growth)": 0.10,
    "Remain stable / unchanged": 0.0,
    "Decrease": -0.10,
}
def compute_consumer_growth(survey_df: pd.DataFrame, column: str = "volume_change",
                            weights: str = None, midpoints: dict = None) -> float:
    """Survey-implied relative change g_c (e.g. 0.107 = +10.7%)."""
    if survey_df is None or survey_df.empty or column not in survey_df:
        return 0.0
    midpoints = midpoints or VOLUME_CHANGE_MIDPOINTS
    g = survey_df[column].map(midpoints)

    if weights and weights in survey_df:
        w = pd.to_numeric(survey_df[weights], errors="coerce")
        # Only rows with BOTH a recognised answer and a weight count. Before, rows
        # with an unrecognised answer were dropped from the numerator but their
        # weight stayed in the denominator, biasing g_c toward zero.
        ok = g.notna() & w.notna()
        if ok.any() and w[ok].sum() > 0:
            return float((g[ok] * w[ok]).sum() / w[ok].sum())
        return 0.0

    g = g.dropna()
    return float(g.mean()) if len(g) else 0.0


def compute_hybrid_demand(
    consumer_demand: float,
    historical_forecast: float,
    alpha: float,
    scenario: float,
) -> float:
    """Blend consumer and historical forecasts, then apply a scenario multiplier.

    D_hybrid = [alpha * D_c + (1 - alpha) * D_h] * scenario.
    Alpha is clipped to [0, 1]; negative final demand is clipped to zero.
    """
    consumer_demand = float(consumer_demand)
    historical_forecast = float(historical_forecast)
    alpha = float(min(1.0, max(0.0, alpha)))
    scenario = float(scenario)

    if not all(math.isfinite(value) for value in (
        consumer_demand, historical_forecast, alpha, scenario
    )):
        raise ValueError("Hybrid-demand inputs must all be finite numbers.")
    if scenario < 0:
        raise ValueError("Scenario multiplier must be non-negative.")

    blended_demand = (
        alpha * consumer_demand
        + (1.0 - alpha) * historical_forecast
    )
    return max(0.0, blended_demand * scenario)

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


def _normalize_series(s: pd.Series, higher_is_better: bool, ref_mask=None) -> pd.Series:
    """Min-max normalize a metric to 0-1 (clipped). `ref_mask` selects which rows
    define the min/max (we use the feasible suppliers, so a supplier that is
    already ruled out can't stretch the scale and distort everyone else's
    score). Falls back to a neutral 1.0 when all reference values are equal."""
    s = s.astype(float)
    ref = s[ref_mask] if ref_mask is not None and ref_mask.any() else s
    lo, hi = ref.min(), ref.max()
    if hi - lo < 1e-9:
        return pd.Series(1.0, index=s.index)
    scaled = ((s - lo) / (hi - lo)).clip(0.0, 1.0)
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
        empty["Feasible"] = pd.Series(dtype=bool)
        empty["Score"] = pd.Series(dtype=float)
        return empty, empty

    full = apply_feasibility(supplier_df, hybrid_demand)

    for col in ["Unit Price", "Lead Time (d)", "Quality %", "Delay Rate %",
                "Defect Rate %", "Reliability %", "Capacity", "Transport Cost"]:
        full[col] = pd.to_numeric(full[col], errors="coerce").fillna(0.0)

    ref = full["Feasible"]  # scale every metric against feasible suppliers only

    cost_score = _normalize_series(full["Unit Price"] + full["Transport Cost"], False, ref)
    quality_score = _normalize_series(full["Quality %"], True, ref)
    # Delivery: normalise lead time and delay rate separately, then average.
    # (Summing percentile ranks threw away the actual size of the differences.)
    delivery_score = (
        _normalize_series(full["Lead Time (d)"], False, ref)
        + _normalize_series(full["Delay Rate %"], False, ref)
    ) / 2.0
    reliability_adj = full["Reliability %"] - full["Defect Rate %"]
    reliability_score = _normalize_series(reliability_adj, True, ref)
    availability_score = _normalize_series(full["Capacity"] - float(hybrid_demand), True, ref)

    # Weights should sum to 1; renormalise in case they drifted or were all zero.
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in WEIGHT_KEYS}
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()} if total_w > 1e-9 else default_weights()

    full["Score"] = (
        w["cost"] * cost_score
        + w["quality"] * quality_score
        + w["delivery"] * delivery_score
        + w["reliability"] * reliability_score
        + w["availability"] * availability_score
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