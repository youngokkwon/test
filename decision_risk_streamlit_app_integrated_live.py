import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Decision Risk Lab Integrated", layout="wide")

FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"
METALS_BASE = "https://api.metals.dev/v1"

st.title("Decision Risk Lab - Integrated Risk Management App")
st.caption("Workbook + FX API + Metals API + Scenario Reflection")


# -------------------------
# Workbook loader
# -------------------------
default_path = Path(__file__).with_name("decision_risk_mvp_workbook.xlsx")
xlsx_path = st.sidebar.text_input("Workbook path", str(default_path))


@st.cache_data
def load_sheets(path):
    xl = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}


def pick_sheet(sheets, candidates):
    for c in candidates:
        if c in sheets:
            return c
    return None


def latest_by_date(df, date_col="date"):
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    max_date = out[date_col].max()
    return out[out[date_col] == max_date]


# -------------------------
# FX API
# -------------------------
@st.cache_data(ttl=1800)
def fetch_fx_latest(base: str, quote: str) -> float:
    url = f"{FRANKFURTER_BASE}/latest"
    params = {"base": base, "symbols": quote}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return float(data["rates"][quote])


@st.cache_data(ttl=3600)
def fetch_fx_timeseries(base: str, quote: str, days: int) -> pd.DataFrame:
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    url = f"{FRANKFURTER_BASE}/{start_date.isoformat()}..{end_date.isoformat()}"
    params = {"base": base, "symbols": quote}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for d, v in data.get("rates", {}).items():
        if quote in v:
            rows.append({"date": pd.to_datetime(d), "rate": float(v[quote])})
    return pd.DataFrame(rows).sort_values("date")


# -------------------------
# Metals API
# -------------------------
@st.cache_data(ttl=1800)
def fetch_metals_current(api_key: str, currency: str = "USD", unit: str = "toz") -> pd.DataFrame:
    url = f"{METALS_BASE}/latest"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"currency": currency, "unit": unit}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rates = data.get("metals") or data.get("rates") or data.get("data") or {}
    rows = []
    for k, v in rates.items():
        try:
            rows.append({"metal": str(k).upper(), "price": float(v)})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_data(ttl=1800)
def fetch_metals_historical(api_key: str, metal: str, days: int, currency: str = "USD", unit: str = "toz") -> pd.DataFrame:
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    url = f"{METALS_BASE}/timeseries"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "currency": currency,
        "unit": unit,
        "metals": metal.lower(),
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        series = data.get("rates") or data.get("timeseries") or {}
        rows = []
        for d, daily in series.items():
            if isinstance(daily, dict):
                val = daily.get(metal.lower()) or daily.get(metal.upper())
                if val is not None:
                    rows.append({"date": pd.to_datetime(d), "price": float(val)})
        df = pd.DataFrame(rows).sort_values("date")
        if not df.empty:
            return df
    except Exception:
        pass

    demo_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    base_map = {"XAU": 2350, "XAG": 30, "XPT": 980, "XPD": 1020}
    base_price = base_map.get(metal.upper(), 100)
    return pd.DataFrame({
        "date": demo_dates,
        "price": [base_price + (i % 7) * 2 - (i % 5) for i in range(len(demo_dates))]
    })


def metal_to_workbook_commodity(metal_code: str) -> str:
    mapping = {
        "XAU": "AU",
        "XAG": "AG",
        "XPT": "XPT",
        "XPD": "XPD",
    }
    return mapping.get(metal_code.upper(), metal_code.upper())


def compute_api_delta_pct(series_df: pd.DataFrame) -> float:
    if series_df is None or len(series_df) < 2:
        return 0.0
    first_val = float(series_df["price" if "price" in series_df.columns else "rate"].iloc[0])
    last_val = float(series_df["price" if "price" in series_df.columns else "rate"].iloc[-1])
    if first_val == 0:
        return 0.0
    return (last_val / first_val - 1.0) * 100.0


def get_price_column(df: pd.DataFrame):
    for c in ["price_usd_std", "price_value", "price_current"]:
        if c in df.columns:
            return c
    return None


# -------------------------
# Load workbook
# -------------------------
try:
    sheets = load_sheets(xlsx_path)
except Exception as e:
    st.error(f"Could not load workbook: {e}")
    st.stop()

scenario_sheet = pick_sheet(sheets, ["scenarios"])
calc_sheet = pick_sheet(sheets, ["calc_results"])
product_sheet = pick_sheet(sheets, ["products"])
commodity_sheet = pick_sheet(sheets, ["commodity_price", "commodity_prices", "commodity_market_clean"])
energy_sheet = pick_sheet(sheets, ["energy_price", "energy_prices", "energy_market_clean"])
fx_sheet = pick_sheet(sheets, ["fx_rate", "fx_rates", "fx_rate_daily"])
risk_sheet = pick_sheet(sheets, ["country_risk", "country_risk_index"])
reg_sheet = pick_sheet(sheets, ["regulation_events", "regulation_data"])

if scenario_sheet is None or calc_sheet is None:
    st.error("Workbook must contain at least 'scenarios' and 'calc_results' sheets.")
    st.stop()

scenario_df = sheets[scenario_sheet].copy()
calc_all = sheets[calc_sheet].copy()
products = sheets[product_sheet].copy() if product_sheet else pd.DataFrame()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Scenario controls")

scenario_names = scenario_df["scenario_name"].dropna().tolist()
selected_scenario_name = st.sidebar.selectbox(
    "Workbook scenario",
    scenario_names,
    index=1 if len(scenario_names) > 1 else 0
)
selected_scenario_id = int(
    scenario_df.loc[scenario_df["scenario_name"] == selected_scenario_name, "scenario_id"].iloc[0]
)

fx_base = st.sidebar.selectbox("FX base", ["USD", "EUR", "JPY", "CNY", "GBP"], index=0)
fx_quote = st.sidebar.selectbox("FX quote", ["KRW", "USD", "EUR", "JPY", "CNY"], index=0)
fx_days = st.sidebar.slider("FX lookback days", 7, 180, 30, 1)

metals_api_key = st.sidebar.text_input(
    "Metals.dev API Key",
    value=os.getenv("METALS_API_KEY", ""),
    type="password"
)
metal = st.sidebar.selectbox("Metal feed", ["XAU", "XAG", "XPT", "XPD"], index=0)
metal_days = st.sidebar.slider("Metal lookback days", 7, 180, 30, 1)

apply_fx_live = st.sidebar.checkbox("Reflect FX API into scenario", value=True)
apply_metal_live = st.sidebar.checkbox("Reflect metal API into scenario", value=True)
manual_extra_metal_shock = st.sidebar.slider("Manual extra metal shock %", -30, 100, 0, 1)

# -------------------------
# Pull live API data
# -------------------------
fx_error = None
metals_error = None

try:
    fx_latest = fetch_fx_latest(fx_base, fx_quote)
    fx_hist = fetch_fx_timeseries(fx_base, fx_quote, fx_days)
    fx_api_delta_pct = compute_api_delta_pct(fx_hist)
except Exception as e:
    fx_latest = None
    fx_hist = pd.DataFrame()
    fx_api_delta_pct = 0.0
    fx_error = str(e)

try:
    if metals_api_key:
        metals_current = fetch_metals_current(metals_api_key, currency="USD", unit="toz")
    else:
        metals_current = pd.DataFrame({"metal": ["XAU", "XAG", "XPT", "XPD"], "price": [2354.2, 29.8, 982.4, 1018.6]})
    metal_hist = fetch_metals_historical(metals_api_key if metals_api_key else "", metal, metal_days, currency="USD", unit="toz")
    metal_api_delta_pct = compute_api_delta_pct(metal_hist) + manual_extra_metal_shock
except Exception as e:
    metals_current = pd.DataFrame()
    metal_hist = pd.DataFrame()
    metal_api_delta_pct = manual_extra_metal_shock
    metals_error = str(e)

# -------------------------
# Prepare calc data
# -------------------------
calc = calc_all.copy()
if "scenario_id" in calc.columns:
    calc = calc[calc["scenario_id"] == selected_scenario_id].copy()

product_name = products["product_name"].iloc[0] if (not products.empty and "product_name" in products.columns) else "Product"

# Baseline workbook deltas
base_fx_delta_pct = 0.0
if "fx_delta_pct" in scenario_df.columns:
    base_fx_delta_pct = float(
        scenario_df.loc[scenario_df["scenario_id"] == selected_scenario_id, "fx_delta_pct"].fillna(0).iloc[0]
    )

final_fx_delta_pct = base_fx_delta_pct + (fx_api_delta_pct if apply_fx_live else 0.0)

# Apply scenario reflection to calc_results if relevant columns exist
calc_reflected = calc.copy()

if apply_fx_live and "total_scenario_cost" in calc_reflected.columns:
    calc_reflected["total_scenario_cost"] = calc_reflected["total_scenario_cost"] * (1 + final_fx_delta_pct / 100.0)

workbook_commodity_code = metal_to_workbook_commodity(metal)
if apply_metal_live and not calc_reflected.empty:
    target_mask = pd.Series([False] * len(calc_reflected))
    if "commodity_code" in calc_reflected.columns:
        target_mask = calc_reflected["commodity_code"].astype(str).str.upper().eq(workbook_commodity_code)
    # if target commodity exists, reflect directly into scenario material / total cost
    if target_mask.any():
        if "scenario_material_cost" in calc_reflected.columns:
            calc_reflected.loc[target_mask, "scenario_material_cost"] = calc_reflected.loc[target_mask, "scenario_material_cost"] * (1 + metal_api_delta_pct / 100.0)
        if "total_scenario_cost" in calc_reflected.columns:
            calc_reflected.loc[target_mask, "total_scenario_cost"] = calc_reflected.loc[target_mask, "total_scenario_cost"] * (1 + metal_api_delta_pct / 100.0)
    else:
        # if workbook does not include precious metal commodity, apply a small proxy shock to total scenario cost
        if "total_scenario_cost" in calc_reflected.columns:
            proxy_factor = 1 + (metal_api_delta_pct / 100.0) * 0.10
            calc_reflected["total_scenario_cost"] = calc_reflected["total_scenario_cost"] * proxy_factor

# Recompute KPI-style metrics
total_base = float(calc_reflected["total_base_cost"].sum()) if "total_base_cost" in calc_reflected.columns else 0
total_scenario = float(calc_reflected["total_scenario_cost"].sum()) if "total_scenario_cost" in calc_reflected.columns else 0
delta_pct = (total_scenario / total_base - 1) if total_base else 0
avg_risk = float(calc_reflected["country_total_risk"].mean()) if ("country_total_risk" in calc_reflected.columns and not calc_reflected.empty) else 0
avg_delay = float(calc_reflected["delay_months"].mean()) if ("delay_months" in calc_reflected.columns and not calc_reflected.empty) else 0

# -------------------------
# Top KPI row
# -------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Product", product_name)
k2.metric("Scenario Cost", f"{total_scenario:,.2f}")
k3.metric("Cost Delta", f"{delta_pct:.1%}")
k4.metric("Avg Risk Score", f"{avg_risk:,.1f}")

st.divider()

# -------------------------
# Live macro KPI row
# -------------------------
mcols = st.columns(4)
mcols[0].metric(f"{fx_base}/{fx_quote}", f"{fx_latest:,.2f}" if fx_latest else "N/A", f"{fx_api_delta_pct:+.2f}%")
sel_metal_df = metals_current[metals_current["metal"].astype(str).str.upper() == metal] if not metals_current.empty else pd.DataFrame()
sel_metal_price = float(sel_metal_df["price"].iloc[0]) if not sel_metal_df.empty else None
mcols[1].metric(f"{metal} Current", f"{sel_metal_price:,.2f}" if sel_metal_price else "N/A", f"{metal_api_delta_pct:+.2f}%")
mcols[2].metric("Reflected FX Delta", f"{final_fx_delta_pct:+.2f}%")
mcols[3].metric("Reflected Metal Delta", f"{metal_api_delta_pct:+.2f}%")

if fx_error:
    st.warning(f"FX API issue: {fx_error}")
if metals_error:
    st.warning(f"Metals API issue: {metals_error}")

st.divider()

left, right = st.columns([1.2, 1])

with left:
    st.subheader("6-month cost path (with live FX/metal reflection)")
    if total_base:
        months = list(range(1, 7))
        curve = []
        for m in months:
            progress = min(1.0, m / max(1.0, avg_delay if avg_delay > 0 else 3))
            curve.append(total_base + (total_scenario - total_base) * progress)
        chart_df = pd.DataFrame({"Month": months, "Base": [total_base] * 6, "Scenario": curve}).set_index("Month")
        st.line_chart(chart_df)

    st.subheader("Part-level scenario impact")
    if {"part_name", "total_base_cost", "total_scenario_cost"}.issubset(calc_reflected.columns):
        bar_df = calc_reflected[["part_name", "total_base_cost", "total_scenario_cost"]].set_index("part_name")
        st.bar_chart(bar_df)

    st.subheader("Detailed calculation table")
    show_cols = [c for c in [
        "part_name", "commodity_code", "base_material_cost", "scenario_material_cost",
        "total_base_cost", "total_scenario_cost", "country_total_risk", "delay_months", "cost_delta_pct"
    ] if c in calc_reflected.columns]
    if show_cols:
        view_df = calc_reflected[show_cols].copy()
        if "cost_delta_pct" in view_df.columns:
            view_df["cost_delta_pct"] = (pd.to_numeric(view_df["cost_delta_pct"], errors="coerce") * 100).round(2)
        st.dataframe(view_df, use_container_width=True)

with right:
    st.subheader("Workbook + live indicator summary")
    summary_rows = [
        ("Workbook scenario", selected_scenario_name),
        ("FX API delta %", round(fx_api_delta_pct, 2)),
        ("Metal API delta %", round(metal_api_delta_pct, 2)),
        ("Final reflected FX delta %", round(final_fx_delta_pct, 2)),
        ("Scenario total cost", round(total_scenario, 2)),
        ("Scenario delta %", round(delta_pct * 100, 2)),
        ("Average delay months", round(avg_delay, 2)),
    ]
    st.table(pd.DataFrame(summary_rows, columns=["Metric", "Value"]))

    st.subheader("Recommended action")
    if delta_pct > 0.12:
        rec = "Switch supplier + partial pass-through"
    elif delta_pct > 0.05:
        rec = "Monitor + partial pass-through"
    else:
        rec = "Hold / monitor"
    st.success(rec)

    st.subheader("Live market tabs")
    tab_names = ["FX", "Metal", "Workbook Commodity", "Risk"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        if not fx_hist.empty:
            st.line_chart(fx_hist.set_index("date")["rate"])
            st.dataframe(fx_hist.tail(10), use_container_width=True)

    with tabs[1]:
        if not metal_hist.empty:
            st.line_chart(metal_hist.set_index("date")["price"])
            st.dataframe(metal_hist.tail(10), use_container_width=True)
        if not metals_current.empty:
            st.dataframe(metals_current, use_container_width=True)

    with tabs[2]:
        if commodity_sheet:
            cdf = sheets[commodity_sheet].copy()
            if "date" in cdf.columns:
                cdf = latest_by_date(cdf, "date")
            st.dataframe(cdf, use_container_width=True)
        else:
            st.info("No commodity workbook sheet found.")

    with tabs[3]:
        if risk_sheet:
            rdf = sheets[risk_sheet].copy()
            if "date" in rdf.columns:
                rdf = latest_by_date(rdf, "date")
            st.dataframe(rdf, use_container_width=True)
        else:
            st.info("No risk sheet found.")

st.divider()

if reg_sheet:
    st.subheader("Regulation / export control events")
    reg_df = sheets[reg_sheet].copy()
    show_reg = [c for c in ["date", "country_code", "commodity_code", "risk_type", "restriction_type", "severity", "title"] if c in reg_df.columns]
    if show_reg:
        st.dataframe(reg_df[show_reg], use_container_width=True)
    else:
        st.dataframe(reg_df, use_container_width=True)

st.subheader("Raw workbook sheet viewer")
sheet_name = st.selectbox("View sheet", list(sheets.keys()))
st.dataframe(sheets[sheet_name], use_container_width=True)
