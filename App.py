"""
Supplier Decision Console — Streamlit prototype (ML-Enabled)
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd

import Logic as L

st.set_page_config(page_title="Supplier Decision Console", layout="wide", page_icon="📦")

# ---------------------------------------------------------------------------
# Styling — dark, data-console look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.stMetric label, .mono { font-family: 'IBM Plex Mono', monospace !important; }

.kicker {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #8A94A1;
    letter-spacing: .02em;
}
.sec-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #E2A23A;
    border: 1px solid #7A5C28;
    padding: 2px 8px;
    border-radius: 3px;
    display: inline-block;
}
.foot-note {
    font-size: 12.5px;
    color: #8A94A1;
    line-height: 1.6;
    padding: 10px 2px;
}
.readout-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 34px;
    font-weight: 600;
    color: #E2A23A;
}
.readout-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #5C6470;
    text-transform: uppercase;
}
.rank-card {
    border: 1px solid #2A323C;
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 10px;
    background: #181D24;
}
.rank-card.winner {
    border-color: #E2A23A;
    background: linear-gradient(180deg, rgba(226,162,58,0.09), rgba(226,162,58,0.02));
}
.recommend-panel {
    border: 1px solid #7A5C28;
    border-radius: 4px;
    background: #1E252D;
    padding: 20px 22px;
    margin-top: 8px;
}
.recommend-panel.empty { border-color: #4A2B27; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "product_id": "P-001",
        "horizon_days": 30,
        "unit_label": "units",
        "consumer_df": L.sample_consumer_df(),
        "historical_df": L.sample_historical_df(),
        "supplier_df": L.sample_supplier_df(),
        "alpha": 0.4,
        "scenario": 1.0,
        "weights": L.default_weights(),
        # --- multi-product support ---
        "consumer_raw_df": None,
        "historical_raw_df": None,
        "supplier_raw_df": None,
        "available_products": [],
        "selected_product": "All products",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    for k in L.WEIGHT_KEYS:
        slider_key = f"slider_{k}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state.weights[k]


def refresh_available_products():
    """Re-scan whichever raw uploads are present for a Product ID column and
    rebuild the master product list shown in the selector."""
    found = set()
    for raw_key in ("consumer_raw_df", "historical_raw_df", "supplier_raw_df"):
        raw = st.session_state.get(raw_key)
        if raw is not None:
            found.update(L.list_products(raw))
    st.session_state.available_products = sorted(found)
    if st.session_state.selected_product != "All products" and \
       st.session_state.selected_product not in st.session_state.available_products:
        st.session_state.selected_product = "All products"


def refresh_product_filtered_tables():
    """Recompute consumer_df / historical_df / supplier_df for whichever
    raw datasets exist, filtered to the currently selected product (or
    combined across all products if 'All products' is selected). Datasets
    with no Product ID column of their own are left untouched -- they stay
    global regardless of the product selector."""
    selection = st.session_state.selected_product
    product_filter = None if selection == "All products" else selection

    supplier_raw = st.session_state.get("supplier_raw_df")
    if supplier_raw is not None:
        st.session_state.supplier_df = L.derive_supplier_table_from_raw(supplier_raw, product_filter)

    for raw_key, df_key, parse_fn in [
        ("consumer_raw_df", "consumer_df", L.parse_consumer_upload),
        ("historical_raw_df", "historical_df", L.parse_historical_upload),
    ]:
        raw = st.session_state.get(raw_key)
        if raw is None:
            continue
        product_col = L.get_product_column(raw)
        if product_col is not None and product_filter is not None:
            # A specific product was chosen: filter straight down to its own rows.
            filtered = raw[raw[product_col].astype(str) == str(product_filter)]
        elif product_col is not None and df_key == "historical_df" and raw[product_col].nunique() > 1:
            # "All products" + a time series that actually covers multiple
            # products: combine by summing demand per date rather than
            # feeding the regression duplicate, unaligned rows per date.
            filtered = L.aggregate_historical_across_products(raw)
        else:
            filtered = raw
        st.session_state[df_key] = parse_fn(filtered)


def on_product_change():
    st.session_state.selected_product = st.session_state["product_selector"]
    refresh_product_filtered_tables()


def clear_all():
    st.session_state.consumer_df = pd.DataFrame(columns=L.CONSUMER_COLUMNS)
    st.session_state.historical_df = pd.DataFrame(columns=L.HISTORICAL_COLUMNS)
    st.session_state.supplier_df = pd.DataFrame(columns=L.SUPPLIER_COLUMNS)
    st.session_state.consumer_raw_df = None
    st.session_state.historical_raw_df = None
    st.session_state.supplier_raw_df = None
    st.session_state.available_products = []
    st.session_state.selected_product = "All products"


def load_sample():
    st.session_state.consumer_df = L.sample_consumer_df()
    st.session_state.historical_df = L.sample_historical_df()
    st.session_state.supplier_df = L.sample_supplier_df()
    st.session_state.alpha = 0.4
    st.session_state.weights = L.default_weights()
    for k in L.WEIGHT_KEYS:
        st.session_state[f"slider_{k}"] = 0.2
    # Sample data has no per-product breakdown -- fall back to a single global view
    st.session_state.consumer_raw_df = None
    st.session_state.historical_raw_df = None
    st.session_state.supplier_raw_df = None
    st.session_state.available_products = []
    st.session_state.selected_product = "All products"


init_state()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="kicker">supplier decision support · ml-enabled build</div>', unsafe_allow_html=True)
st.title("Supplier Decision Console")
st.caption(
    "Enter consumer survey data, historical demand, and current supplier terms below. "
    "The console uses an XGBoost regressor for survey behavior and an LSTM network for time-series forecasting, "
    "evaluating supplier feasibility and ranking options based on multi-criteria weights."
)

with st.container(border=True):
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1.2, 1, 1])
    with c1:
        st.session_state.product_id = st.text_input("Product ID (label)", st.session_state.product_id)
    with c2:
        st.session_state.horizon_days = st.number_input(
            "Forecast horizon (days)", min_value=1, value=int(st.session_state.horizon_days)
        )
    with c3:
        st.session_state.unit_label = st.text_input("Unit", st.session_state.unit_label)
    with c4:
        product_options = ["All products"] + st.session_state.available_products
        if st.session_state.selected_product not in product_options:
            st.session_state.selected_product = "All products"
        st.selectbox(
            "Product filter",
            product_options,
            index=product_options.index(st.session_state.selected_product),
            key="product_selector",
            on_change=on_product_change,
            help="Populated automatically once you upload a dataset that has a Product ID column. "
                 "Selecting a product re-derives supplier metrics (and any other uploaded dataset with "
                 "its own Product ID column) from just that product's data.",
        )
    with c5:
        st.write("")
        st.button("Load sample data", on_click=load_sample, width='stretch')
    with c6:
        st.write("")
        st.button("Clear all", on_click=clear_all, width='stretch')

if st.session_state.available_products:
    st.caption(
        f"📦 Product-aware mode: {len(st.session_state.available_products)} product(s) detected "
        f"({', '.join(st.session_state.available_products)}). Currently viewing: "
        f"**{st.session_state.selected_product}**."
    )

st.divider()

# ---------------------------------------------------------------------------
# 01 — Consumer survey data (XGBoost)
# ---------------------------------------------------------------------------
st.markdown('<span class="sec-num">01 · consumer behavior model (XGBoost)</span>', unsafe_allow_html=True)
st.subheader("Consumer survey data")
st.caption("Each row represents a survey segment. Behind the scenes, an XGBoost model evaluates categorical attributes.")

if st.session_state.consumer_raw_df is not None and L.get_product_column(st.session_state.consumer_raw_df):
    st.info(
        f"Filtered to **{st.session_state.selected_product}** "
        f"({len(st.session_state.consumer_df)} of {len(st.session_state.consumer_raw_df)} survey rows). "
        f"Switch the Product filter above to recompute for a different product.",
        icon="🔄",
    )

col_left, col_right = st.columns([1.5, 1])

with col_left:
    up = st.file_uploader("Upload consumer survey (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="consumer_upload")
    if up is not None:
        try:
            raw = L.read_any_table(up)
            st.session_state.consumer_raw_df = raw
            refresh_available_products()
            refresh_product_filtered_tables()
            product_col = L.get_product_column(raw)
            note = f" (Product ID column detected: '{product_col}')" if product_col else ""
            st.success(f"Imported {len(st.session_state.consumer_df)} consumer row(s) from {up.name}.{note}")
        except Exception as e:
            st.error(f"Could not read that file: {e}")

    st.download_button(
        "Download template", data=L.template_bytes("consumer"),
        file_name="consumer_survey_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    edited_consumer = st.data_editor(
        st.session_state.consumer_df,
        num_rows="dynamic",
        width='stretch',
        key="consumer_editor",
        column_config={
            "Purchase Frequency": st.column_config.SelectboxColumn(options=L.LEVELS),
            "Price Sensitivity": st.column_config.SelectboxColumn(options=L.LEVELS),
            "Promotion Response": st.column_config.SelectboxColumn(options=L.LEVELS),
            "Satisfaction": st.column_config.SelectboxColumn(options=L.LEVELS),
            "Repurchase Intention": st.column_config.SelectboxColumn(options=L.LEVELS),
            "Previous Purchase": st.column_config.NumberColumn(min_value=0),
            "Represents": st.column_config.NumberColumn(min_value=1),
        },
    )
    st.session_state.consumer_df = edited_consumer

with col_right:
    consumer_demand = L.compute_consumer_demand(st.session_state.consumer_df)
    st.markdown(f"""
    <div class="readout-label">consumer-based demand (XGBoost)</div>
    <div class="readout-value">{consumer_demand:,.0f}<span style="font-size:16px;color:#8A94A1;"> {st.session_state.unit_label}</span></div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="foot-note"><b>ML Architecture.</b> Categorical dimensions are one-hot encoded and passed '
        'into an XGBoost regressor that calculates non-linear interaction weights between customer habits, price tiers, '
        'and promotional responsiveness.</div>', unsafe_allow_html=True
    )

st.divider()

# ---------------------------------------------------------------------------
# 02 — Historical demand data (LSTM)
# ---------------------------------------------------------------------------
st.markdown('<span class="sec-num">02 · time-series model (LSTM Neural Network)</span>', unsafe_allow_html=True)
st.subheader("Historical demand data")
st.caption("Past demand observations are processed through a sliding window sequence into an LSTM network.")

if st.session_state.historical_raw_df is not None and L.get_product_column(st.session_state.historical_raw_df):
    n_products = st.session_state.historical_raw_df[L.get_product_column(st.session_state.historical_raw_df)].nunique()
    if st.session_state.selected_product == "All products":
        st.info(
            f"Showing **combined demand summed across all {n_products} products** "
            f"({len(st.session_state.historical_df)} daily totals). This is not the same as any single "
            f"product's series — it's the whole business's demand on each date.",
            icon="🔄",
        )
    else:
        st.info(
            f"Filtered to **{st.session_state.selected_product}** only "
            f"({len(st.session_state.historical_df)} of {len(st.session_state.historical_raw_df)} records).",
            icon="🔄",
        )

col_left, col_right = st.columns([1.5, 1])

with col_left:
    up = st.file_uploader("Upload historical demand (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="historical_upload")
    if up is not None:
        try:
            raw = L.read_any_table(up)
            st.session_state.historical_raw_df = raw
            refresh_available_products()
            refresh_product_filtered_tables()
            if st.session_state.historical_df.empty:
                st.error("Found rows, but couldn't read a Date column.")
            else:
                product_col = L.get_product_column(raw)
                note = f" (Product ID column detected: '{product_col}')" if product_col else ""
                st.success(f"Imported {len(st.session_state.historical_df)} demand record(s) from {up.name}.{note}")
        except Exception as e:
            st.error(f"Could not read that file: {e}")

    st.download_button(
        "Download template", data=L.template_bytes("historical"),
        file_name="historical_demand_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    edited_hist = st.data_editor(
        st.session_state.historical_df,
        num_rows="dynamic",
        width='stretch',
        key="historical_editor",
        column_config={
            "Date": st.column_config.TextColumn(help="YYYY-MM-DD"),
            "Demand": st.column_config.NumberColumn(min_value=0),
            "Price": st.column_config.NumberColumn(min_value=0),
            "Seasonal Factor": st.column_config.NumberColumn(min_value=0, step=0.01),
            "Promotion": st.column_config.CheckboxColumn(),
        },
    )
    st.session_state.historical_df = edited_hist

with col_right:
    historical_forecast, ok, avg_step = L.compute_historical_forecast(
        st.session_state.historical_df, st.session_state.horizon_days
    )
    st.markdown(f"""
    <div class="readout-label">historical trend forecast (LSTM)</div>
    <div class="readout-value">{historical_forecast:,.0f}<span style="font-size:16px;color:#8A94A1;"> {st.session_state.unit_label}</span></div>
    """, unsafe_allow_html=True)
    sub = (f"LSTM sequence trained across {len(st.session_state.historical_df)} record(s), "
           f"projected {st.session_state.horizon_days} days ahead") if ok else "Add at least two dated records"
    st.caption(sub)
    st.markdown(
        '<div class="foot-note"><b>ML Architecture.</b> MinMax scaling normalizes historical values before '
        'passing them into a recurrent LSTM block to extrapolate non-linear temporal trends across the horizon.</div>', unsafe_allow_html=True
    )

st.divider()

# ---------------------------------------------------------------------------
# 03 — Hybrid forecast
# ---------------------------------------------------------------------------
st.markdown('<span class="sec-num">03 · D_hybrid = &alpha;D_c + (1&minus;&alpha;)D_h</span>', unsafe_allow_html=True)
st.subheader("Hybrid demand forecast")
st.caption("Blend the ML outputs together and stress-test the combined baseline against a demand scenario.")

col_left, col_right = st.columns([1.5, 1])

with col_left:
    st.session_state.alpha = st.slider(
        "Weight on consumer-based demand (α)", 0.0, 1.0, float(st.session_state.alpha), step=0.05
    )
    st.caption(f"Historical weight is set automatically to 1 − α = {1 - st.session_state.alpha:.2f}")

    scenario_label = st.radio(
        "Demand scenario", ["Low demand · ×0.8", "Normal · ×1.0", "High demand · ×1.2"],
        index=1, horizontal=True,
    )
    st.session_state.scenario = {"Low demand · ×0.8": 0.8, "Normal · ×1.0": 1.0, "High demand · ×1.2": 1.2}[scenario_label]

with col_right:
    hybrid_demand = L.compute_hybrid_demand(
        consumer_demand, historical_forecast, st.session_state.alpha, st.session_state.scenario
    )
    st.markdown(f"""
    <div class="readout-label">final forecasted demand</div>
    <div class="readout-value">{hybrid_demand:,.0f}<span style="font-size:16px;color:#8A94A1;"> {st.session_state.unit_label}</span></div>
    """, unsafe_allow_html=True)
    st.caption(
        f"α={st.session_state.alpha:.2f} · D = {st.session_state.alpha:.2f}({consumer_demand:,.0f}) + "
        f"{1-st.session_state.alpha:.2f}({historical_forecast:,.0f}) × scenario {st.session_state.scenario}"
    )

st.divider()

# ---------------------------------------------------------------------------
# 04 — Supplier data
# ---------------------------------------------------------------------------
st.markdown('<span class="sec-num">04 · feasibility &amp; performance</span>', unsafe_allow_html=True)
st.subheader("Supplier data")
st.caption("Enter current terms for each candidate supplier. Feasibility is checked live against the forecast above.")

if st.session_state.supplier_raw_df is not None:
    st.info(
        f"Showing supplier metrics **derived live** from raw purchase-order history for "
        f"**{st.session_state.selected_product}**. Editing a cell below is a temporary override — "
        f"switching the Product filter above re-derives this table fresh from that product's orders.",
        icon="🔄",
    )

up = st.file_uploader("Upload supplier data (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="supplier_upload")
if up is not None:
    try:
        raw = L.read_any_table(up)
        if L.is_raw_supplier_transactions(raw):
            # Raw per-order purchase history (e.g. PO_ID, Order_Date, Delivery_Date,
            # Quantity, Defective_Units, Compliance, Product_Id...). Store it and
            # derive the supplier table live, per the current product selection.
            st.session_state.supplier_raw_df = raw
            refresh_available_products()
            refresh_product_filtered_tables()
            product_col = L.get_product_column(raw)
            products_note = f" across {len(st.session_state.available_products)} product(s)" if product_col else ""
            st.success(
                f"Imported {len(raw)} raw purchase order(s) from {up.name}{products_note}. "
                f"Supplier metrics below are derived live from this order history."
            )
        else:
            # Legacy path: already one row per supplier.
            parsed = L.parse_supplier_upload(raw)
            st.session_state.supplier_raw_df = None
            if parsed.empty:
                st.error("Couldn't identify supplier columns or rows.")
            else:
                st.session_state.supplier_df = parsed
                st.success(f"Imported {len(parsed)} supplier(s) from {up.name}.")
    except Exception as e:
        st.error(f"Could not read that file: {e}")

st.download_button(
    "Download template", data=L.template_bytes("supplier"),
    file_name="supplier_data_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

edited_supplier = st.data_editor(
    st.session_state.supplier_df,
    num_rows="dynamic",
    width='stretch',
    key="supplier_editor",
    column_config={
        "Unit Price": st.column_config.NumberColumn(min_value=0, format="$%.2f"),
        "Lead Time (d)": st.column_config.NumberColumn(min_value=0),
        "Quality %": st.column_config.NumberColumn(min_value=0, max_value=100),
        "Capacity": st.column_config.NumberColumn(min_value=0),
        "Delay Rate %": st.column_config.NumberColumn(min_value=0, max_value=100),
        "Defect Rate %": st.column_config.NumberColumn(min_value=0, max_value=100),
        "Reliability %": st.column_config.NumberColumn(min_value=0, max_value=100),
        "Transport Cost": st.column_config.NumberColumn(min_value=0, format="$%.2f"),
    },
)
st.session_state.supplier_df = edited_supplier

full_suppliers = L.apply_feasibility(st.session_state.supplier_df, hybrid_demand) if not st.session_state.supplier_df.empty else st.session_state.supplier_df
if not full_suppliers.empty:
    display_df = full_suppliers.copy()
    display_df["Feasible"] = display_df["Feasible"].map({True: "✅ Yes", False: "❌ No"})
    st.dataframe(display_df, width='stretch', hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# 05 — Decision weights & recommendation
# ---------------------------------------------------------------------------
st.markdown('<span class="sec-num">05 · Multi-criteria optimization</span>', unsafe_allow_html=True)
st.subheader("Decision weights & recommendation")
st.caption("Set how much each criterion matters. Only feasible suppliers are scored and ranked.")


def on_weight_change(changed_key):
    new_val = st.session_state[f"slider_{changed_key}"]
    updated = L.redistribute_weights(st.session_state.weights, changed_key, new_val)
    st.session_state.weights = updated
    for k in L.WEIGHT_KEYS:
        st.session_state[f"slider_{k}"] = updated[k]


col_left, col_right = st.columns([1.5, 1])

with col_left:
    w1, w2 = st.columns(2)
    columns_cycle = [w1, w2]
    for i, key in enumerate(L.WEIGHT_KEYS):
        with columns_cycle[i % 2]:
            pct = round(st.session_state.weights[key] * 100)
            st.slider(
                f"{L.WEIGHT_LABELS[key]} — {pct}%", 0.0, 1.0,
                key=f"slider_{key}", step=0.01,
                on_change=on_weight_change, args=(key,),
            )

with col_right:
    st.markdown(
        '<div class="foot-note">'
        '<b>Cost</b> — unit price, lower is better.<br>'
        '<b>Quality</b> — reported quality rate, higher is better.<br>'
        '<b>Delivery</b> — blend of lead time and delay rate, lower is better.<br>'
        '<b>Reliability</b> — reliability rate adjusted down by defect rate, higher is better.<br>'
        '<b>Availability</b> — spare capacity above the forecast, higher is better.'
        '</div>', unsafe_allow_html=True
    )

st.write("")

full_result, ranked = L.compute_ranking(st.session_state.supplier_df, hybrid_demand, st.session_state.weights)

if st.session_state.supplier_df.empty:
    st.info("Add at least one supplier to see a recommendation.")
elif ranked.empty:
    st.warning(f"No supplier can cover the forecasted demand of {hybrid_demand:,.0f} {st.session_state.unit_label}.")
else:
    for i, row in ranked.iterrows():
        winner = i == 0
        st.markdown(f"""
        <div class="rank-card {'winner' if winner else ''}">
          <b>{i+1:02d}. {row['Supplier']}</b> &nbsp;
          <span style="font-family:'IBM Plex Mono',monospace;color:#8A94A1;font-size:12px;">
            ${row['Unit Price']:.2f}/unit · {row['Lead Time (d)']:.0f}d lead · {row['Quality %']:.0f}% quality ·
            {row['Capacity']:.0f} capacity
          </span>
          <span style="float:right; font-family:'IBM Plex Mono',monospace; font-size:18px;">{row['Score']*100:.1f}</span>
        </div>
        """, unsafe_allow_html=True)

    top = ranked.iloc[0]
    reasons = L.generate_reasons(top, ranked)
    st.markdown(f"""
    <div class="recommend-panel">
      <div class="sec-num">recommended supplier</div>
      <h3 style="margin-top:10px;">{top['Supplier']}</h3>
      <ul>{"".join(f"<li>{r}</li>" for r in reasons)}</ul>
    </div>
    """, unsafe_allow_html=True)
