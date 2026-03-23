import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="FX + Metals Dashboard", layout="wide")

st.title("FX + Metals Live Dashboard")
st.caption("Frankfurter + Metals.dev 기반 간단 대시보드 (API 키 없으면 금속은 데모 데이터 사용)")

FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"
METALS_BASE = "https://api.metals.dev/v1"


@st.cache_data(ttl=3600)
def fetch_fx_timeseries(base: str, quote: str, days: int) -> pd.DataFrame:
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    url = f"{FRANKFURTER_BASE}/{start_date.isoformat()}..{end_date.isoformat()}"
    params = {"base": base, "symbols": quote}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rates = data.get("rates", {})
    rows = []
    for d, v in rates.items():
        if quote in v:
            rows.append({"date": pd.to_datetime(d), "rate": float(v[quote])})

    return pd.DataFrame(rows).sort_values("date")


@st.cache_data(ttl=1800)
def fetch_fx_latest(base: str, quote: str) -> float:
    url = f"{FRANKFURTER_BASE}/latest"
    params = {"base": base, "symbols": quote}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    return float(data["rates"][quote])


@st.cache_data(ttl=1800)
def fetch_metals_current(api_key: str, currencies: str = "USD", unit: str = "toz") -> pd.DataFrame:
    url = f"{METALS_BASE}/latest"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"currency": currencies, "unit": unit}
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


with st.sidebar:
    st.header("설정")

    st.subheader("환율")
    fx_base = st.selectbox("Base currency", ["USD", "EUR", "JPY", "CNY", "GBP"], index=0)
    fx_quote = st.selectbox("Quote currency", ["KRW", "USD", "EUR", "JPY", "CNY"], index=0)
    fx_days = st.slider("환율 조회 기간 (일)", 7, 180, 30, 1)

    st.subheader("금속")
    metals_api_key = st.text_input(
        "Metals.dev API Key",
        value=os.getenv("METALS_API_KEY", ""),
        type="password",
        help="없으면 금속 현재가/차트는 데모 데이터 또는 일부 실패할 수 있어요."
    )
    metal = st.selectbox("금속", ["XAU", "XAG", "XPT", "XPD"], index=0)
    metal_days = st.slider("금속 조회 기간 (일)", 7, 180, 30, 1)
    unit = st.selectbox("단위", ["toz"], index=0)

st.subheader("1) 환율")
try:
    latest_fx = fetch_fx_latest(fx_base, fx_quote)
    fx_df = fetch_fx_timeseries(fx_base, fx_quote, fx_days)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(f"{fx_base}/{fx_quote}", f"{latest_fx:,.2f}")
        if len(fx_df) >= 2:
            delta = fx_df["rate"].iloc[-1] - fx_df["rate"].iloc[0]
            st.metric("기간 변화", f"{delta:,.2f}")
    with c2:
        st.line_chart(fx_df.set_index("date")["rate"])
        st.dataframe(fx_df.tail(10), use_container_width=True)
except Exception as e:
    st.error(f"환율 데이터 로딩 실패: {e}")

st.divider()

st.subheader("2) 금속 현재가 + 차트")

m1, m2 = st.columns([1, 2])

with m1:
    if metals_api_key:
        try:
            current_metals = fetch_metals_current(metals_api_key, currencies="USD", unit=unit)
            if not current_metals.empty:
                st.dataframe(current_metals, use_container_width=True)
                sel = current_metals[current_metals["metal"] == metal]
                if not sel.empty:
                    st.metric(f"{metal} Current", f'{float(sel["price"].iloc[0]):,.2f} USD/{unit}')
            else:
                st.warning("금속 현재가 데이터를 받았지만 비어 있어요.")
        except Exception as e:
            st.warning(f"현재가 API 호출 실패: {e}")
    else:
        demo_current = pd.DataFrame({
            "metal": ["XAU", "XAG", "XPT", "XPD"],
            "price": [2354.2, 29.8, 982.4, 1018.6]
        })
        st.info("API 키가 없어 데모 현재가를 표시합니다.")
        st.dataframe(demo_current, use_container_width=True)
        sel = demo_current[demo_current["metal"] == metal]
        st.metric(f"{metal} Current (Demo)", f'{float(sel["price"].iloc[0]):,.2f} USD/{unit}')

with m2:
    try:
        if metals_api_key:
            hist_df = fetch_metals_historical(metals_api_key, metal, metal_days, currency="USD", unit=unit)
        else:
            hist_df = fetch_metals_historical("", metal, metal_days, currency="USD", unit=unit)
        st.line_chart(hist_df.set_index("date")["price"])
        st.dataframe(hist_df.tail(10), use_container_width=True)
    except Exception as e:
        st.error(f"금속 차트 로딩 실패: {e}")

st.divider()
st.subheader("3) 활용 팁")
st.markdown(
    """
- 환율은 무료 공개 API로 바로 연결 가능
- 귀금속은 API 키만 있으면 현재가와 차트 구성 가능
- 산업금속(구리/알루미늄/니켈)의 공식 benchmark 실시간 가격은 보통 유상 라이선스 검토가 필요
- 지금 대시보드는 FX + 금속 시세 카드 + 시계열 차트를 빠르게 검증하는 MVP 용도
"""
)
