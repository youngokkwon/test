import io
import uuid
from typing import Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Product Risk Dashboard", layout="wide")

REQUIRED_COLUMNS = [
    "product_id", "part_no", "parent_part_no", "part_name",
    "qty", "material", "weight", "unit_cost",
]
OPTIONAL_COLUMNS = [
    "supplier", "country", "process", "currency", "material_group",
    "ef_mapping_key", "ef_value", "commodity_mapping_key", "commodity_price",
    "fx_exposure", "energy_intensity", "volatility_beta", "supply_risk_score",
]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def init_state():
    defaults = {
        "bom_df": pd.DataFrame(columns=ALL_COLUMNS + ["node_id", "level", "status"]),
        "selected_product_ids": [],
        "selected_part_no": None,
        "scenario_configs": {},
        "upload_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def default_scenario():
    return {
        "scenario_name": "Base Stress",
        "material_default_delta": 0.05,
        "material_group_deltas": {
            "Aluminum": 0.10, "Steel": 0.03, "Plastic": 0.04, "Electronics": 0.07,
        },
        "fx_delta": 0.05,
        "energy_delta": 0.08,
        "logistics_delta": 0.02,
        "supply_risk_multiplier": 0.20,
    }


def sample_bom() -> pd.DataFrame:
    data = [
        {"product_id": "PRD-100", "part_no": "PRD-100", "parent_part_no": "", "part_name": "Motor Assembly", "qty": 1, "material": "", "weight": 0, "unit_cost": 0, "supplier": "", "country": "KR", "process": "", "currency": "KRW", "material_group": "", "ef_mapping_key": "", "ef_value": None, "commodity_mapping_key": "", "commodity_price": None, "fx_exposure": "", "energy_intensity": None, "volatility_beta": None, "supply_risk_score": 0.10},
        {"product_id": "PRD-100", "part_no": "HSG-101", "parent_part_no": "PRD-100", "part_name": "Housing", "qty": 1, "material": "Aluminum ADC12", "weight": 1.5, "unit_cost": 5500, "supplier": "ABC Metal", "country": "CN", "process": "Die Casting", "currency": "KRW", "material_group": "Aluminum", "ef_mapping_key": "aluminum_adc12", "ef_value": 8.7, "commodity_mapping_key": "aluminum", "commodity_price": 2.7, "fx_exposure": "Y", "energy_intensity": 0.8, "volatility_beta": 0.7, "supply_risk_score": 0.45},
        {"product_id": "PRD-100", "part_no": "BLT-102", "parent_part_no": "PRD-100", "part_name": "Bolt Set", "qty": 6, "material": "Steel", "weight": 0.06, "unit_cost": 120, "supplier": "Fasten Co", "country": "KR", "process": "Cold Heading", "currency": "KRW", "material_group": "Steel", "ef_mapping_key": "steel_generic", "ef_value": 2.1, "commodity_mapping_key": "steel", "commodity_price": 0.95, "fx_exposure": "N", "energy_intensity": 0.2, "volatility_beta": 0.3, "supply_risk_score": 0.12},
        {"product_id": "PRD-200", "part_no": "PRD-200", "parent_part_no": "", "part_name": "Control Box", "qty": 1, "material": "", "weight": 0, "unit_cost": 0, "supplier": "", "country": "KR", "process": "", "currency": "KRW", "material_group": "", "ef_mapping_key": "", "ef_value": None, "commodity_mapping_key": "", "commodity_price": None, "fx_exposure": "", "energy_intensity": None, "volatility_beta": None, "supply_risk_score": 0.08},
        {"product_id": "PRD-200", "part_no": "PCB-201", "parent_part_no": "PRD-200", "part_name": "PCB Assy", "qty": 1, "material": "PCB", "weight": 0.4, "unit_cost": 12000, "supplier": "ElecSys", "country": "TW", "process": "SMT", "currency": "KRW", "material_group": "Electronics", "ef_mapping_key": "pcb_generic", "ef_value": 15.0, "commodity_mapping_key": "electronics_proxy", "commodity_price": 8.0, "fx_exposure": "Y", "energy_intensity": 0.6, "volatility_beta": 0.8, "supply_risk_score": 0.33},
        {"product_id": "PRD-200", "part_no": "RSN-202", "parent_part_no": "PRD-200", "part_name": "Resin Cover", "qty": 1, "material": "ABS Resin", "weight": 0.3, "unit_cost": 2500, "supplier": "PolyChem", "country": "TH", "process": "Injection Molding", "currency": "KRW", "material_group": "Plastic", "ef_mapping_key": "plastic_abs", "ef_value": 3.5, "commodity_mapping_key": "petrochemical_proxy", "commodity_price": 1.8, "fx_exposure": "Y", "energy_intensity": 0.5, "volatility_beta": 0.5, "supply_risk_score": 0.22},
        {"product_id": "PRD-300", "part_no": "PRD-300", "parent_part_no": "", "part_name": "Battery Pack", "qty": 1, "material": "", "weight": 0, "unit_cost": 0, "supplier": "", "country": "KR", "process": "", "currency": "KRW", "material_group": "", "ef_mapping_key": "", "ef_value": None, "commodity_mapping_key": "", "commodity_price": None, "fx_exposure": "", "energy_intensity": None, "volatility_beta": None, "supply_risk_score": 0.10},
        {"product_id": "PRD-300", "part_no": "CEL-301", "parent_part_no": "PRD-300", "part_name": "Battery Cell", "qty": 8, "material": "Battery Materials", "weight": 2.4, "unit_cost": 3200, "supplier": "PowerCell", "country": "CN", "process": "Cell Assembly", "currency": "KRW", "material_group": "Electronics", "ef_mapping_key": "battery_cell_generic", "ef_value": 18.0, "commodity_mapping_key": "battery_materials", "commodity_price": 12.5, "fx_exposure": "Y", "energy_intensity": 0.9, "volatility_beta": 0.9, "supply_risk_score": 0.52},
    ]
    return pd.DataFrame(data)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[ALL_COLUMNS].copy()
    for col in ["qty", "weight", "unit_cost", "ef_value", "commodity_price", "energy_intensity", "volatility_beta", "supply_risk_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["part_no"] = df["part_no"].astype(str).str.strip()
    df["parent_part_no"] = df["parent_part_no"].fillna("").astype(str).str.strip()
    df["part_name"] = df["part_name"].astype(str).str.strip()
    df["node_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    df["level"] = 0
    df["status"] = "Normal"
    return build_level_map(df)


def build_level_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parent_map = dict(zip(df["part_no"], df["parent_part_no"]))
    product_map = dict(zip(df["part_no"], df["product_id"]))

    def get_level(part_no: str, visited=None):
        if visited is None:
            visited = set()
        if part_no in visited:
            return 0
        visited.add(part_no)
        parent = str(parent_map.get(part_no, "")).strip()
        if not parent or parent not in parent_map or product_map.get(parent) != product_map.get(part_no):
            return 0
        return 1 + get_level(parent, visited)

    df["level"] = df["part_no"].apply(get_level)
    return df


def children_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    cmap = {}
    for _, row in df.iterrows():
        key = str(row["parent_part_no"]).strip()
        cmap.setdefault(key, []).append(str(row["part_no"]).strip())
    return cmap


def get_product_options(df: pd.DataFrame) -> List[str]:
    return sorted([p for p in df["product_id"].dropna().unique().tolist() if str(p).strip()])


def compute_part_risk_score(row: pd.Series, config: dict) -> float:
    price_beta = float(row.get("volatility_beta", 0) or 0)
    supply_risk = float(row.get("supply_risk_score", 0) or 0)
    fx_exposure = 1.0 if str(row.get("fx_exposure", "")).strip().upper() == "Y" else 0.0
    energy_intensity = float(row.get("energy_intensity", 0) or 0)

    material_delta = config.get("material_default_delta", 0.0)
    material_group = str(row.get("material_group", "")).strip()
    if material_group in config.get("material_group_deltas", {}):
        material_delta = config["material_group_deltas"][material_group]

    risk = (
        abs(material_delta) * (0.35 + price_beta * 0.25)
        + abs(config.get("fx_delta", 0.0)) * (0.2 + fx_exposure * 0.2)
        + abs(config.get("energy_delta", 0.0)) * (0.1 + energy_intensity * 0.15)
        + supply_risk * config.get("supply_risk_multiplier", 0.2)
        + abs(config.get("logistics_delta", 0.0)) * 0.1
    )
    return round(min(risk, 1.0), 4)


def compute_part_costs(row: pd.Series, config: dict):
    qty = float(row.get("qty", 0) or 0)
    unit_cost = float(row.get("unit_cost", 0) or 0)
    base_cost = qty * unit_cost

    material_delta = config.get("material_default_delta", 0.0)
    material_group = str(row.get("material_group", "")).strip()
    if material_group in config.get("material_group_deltas", {}):
        material_delta = config["material_group_deltas"][material_group]

    fx_delta = config.get("fx_delta", 0.0) if str(row.get("fx_exposure", "")).strip().upper() == "Y" else 0.0
    energy_delta = config.get("energy_delta", 0.0) * float(row.get("energy_intensity", 0) or 0)
    logistics_delta = config.get("logistics_delta", 0.0)
    risk_premium = float(row.get("supply_risk_score", 0) or 0) * config.get("supply_risk_multiplier", 0.0)

    scenario_factor = max(1 + material_delta + fx_delta + energy_delta + logistics_delta + risk_premium, 0)
    scenario_cost = base_cost * scenario_factor
    delta_pct = (scenario_cost - base_cost) / base_cost if base_cost > 0 else 0.0
    return round(base_cost, 2), round(scenario_cost, 2), round(delta_pct, 4)


def run_scenario(df: pd.DataFrame, selected_products: List[str]):
    work = df[df["product_id"].isin(selected_products)].copy()
    if work.empty:
        return work, pd.DataFrame()

    results, summaries = [], []

    for product_id in selected_products:
        product_df = work[work["product_id"] == product_id].copy()
        cfg = st.session_state["scenario_configs"].get(product_id, default_scenario())

        product_rows = []
        for _, row in product_df.iterrows():
            base_cost, scenario_cost, delta_pct = compute_part_costs(row, cfg)
            risk_score = compute_part_risk_score(row, cfg)
            risk_status = "High" if risk_score >= 0.60 else ("Medium" if risk_score >= 0.30 else "Low")
            item = row.to_dict()
            item.update({
                "base_cost": base_cost,
                "scenario_cost": scenario_cost,
                "delta_pct": delta_pct,
                "risk_score": risk_score,
                "risk_status": risk_status,
                "scenario_name": cfg["scenario_name"],
            })
            product_rows.append(item)

        part_result_df = pd.DataFrame(product_rows)
        results.append(part_result_df)

        non_root = part_result_df[part_result_df["part_no"] != part_result_df["product_id"]]
        total_base = float(non_root["base_cost"].sum())
        total_scenario = float(non_root["scenario_cost"].sum())
        delta = (total_scenario - total_base) / total_base if total_base > 0 else 0.0
        avg_risk = float(non_root["risk_score"].mean()) if len(non_root) else 0.0
        top_driver = "-"
        if len(non_root):
            top = non_root.sort_values("delta_pct", ascending=False).iloc[0]
            top_driver = f"{top['part_name']} ({round(top['delta_pct']*100,1)}%)"

        summaries.append({
            "product_id": product_id,
            "scenario_name": cfg["scenario_name"],
            "total_base_cost": round(total_base, 2),
            "total_scenario_cost": round(total_scenario, 2),
            "delta_pct": round(delta, 4),
            "avg_risk_score": round(avg_risk, 4),
            "high_risk_parts": int((non_root["risk_status"] == "High").sum()),
            "medium_risk_parts": int((non_root["risk_status"] == "Medium").sum()),
            "top_driver": top_driver,
        })

    return pd.concat(results, ignore_index=True), pd.DataFrame(summaries)


def render_tree(df: pd.DataFrame, product_id: str):
    pdf = df[df["product_id"] == product_id].copy()
    cmap = children_map(pdf)

    def get_row(part_no):
        m = pdf[pdf["part_no"] == part_no]
        return None if m.empty else m.iloc[0]

    def icon(status):
        return {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(status, "⚪")

    def render_node(part_no):
        row = get_row(part_no)
        if row is None:
            return
        label = f"{icon(row.get('risk_status', 'Low'))} {row['part_no']} | {row['part_name']}"
        with st.expander(label, expanded=(row["level"] <= 0)):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.caption(
                    f"Material: {row.get('material','')} | Supplier: {row.get('supplier','')} | "
                    f"Country: {row.get('country','')} | Scenario Cost: {row.get('scenario_cost',0):,.0f}"
                )
            with c2:
                if st.button("선택", key=f"{product_id}_{part_no}"):
                    st.session_state["selected_part_no"] = part_no
                    st.rerun()
            for child in cmap.get(part_no, []):
                render_node(child)

    roots = pdf[pdf["parent_part_no"].astype(str).str.strip() == ""]["part_no"].tolist()
    for root in roots:
        render_node(root)


def scenario_sidebar(df: pd.DataFrame):
    st.sidebar.header("Scenario Settings")
    product_ids = get_product_options(df)
    selected = st.sidebar.multiselect("제품 선택", product_ids, default=product_ids[:2] if product_ids else [])
    st.session_state["selected_product_ids"] = selected

    for product_id in selected:
        st.sidebar.markdown(f"### {product_id}")
        cfg = st.session_state["scenario_configs"].get(product_id, default_scenario()).copy()
        cfg["scenario_name"] = st.sidebar.text_input("Scenario Name", value=cfg["scenario_name"], key=f"sn_{product_id}")
        cfg["material_default_delta"] = st.sidebar.slider("Default Material Δ", -0.5, 1.0, float(cfg["material_default_delta"]), 0.01, key=f"md_{product_id}")
        cfg["fx_delta"] = st.sidebar.slider("FX Δ", -0.5, 1.0, float(cfg["fx_delta"]), 0.01, key=f"fx_{product_id}")
        cfg["energy_delta"] = st.sidebar.slider("Energy Δ", -0.5, 1.0, float(cfg["energy_delta"]), 0.01, key=f"en_{product_id}")
        cfg["logistics_delta"] = st.sidebar.slider("Logistics Δ", -0.5, 1.0, float(cfg["logistics_delta"]), 0.01, key=f"log_{product_id}")
        cfg["supply_risk_multiplier"] = st.sidebar.slider("Supply Risk Multiplier", 0.0, 1.0, float(cfg["supply_risk_multiplier"]), 0.01, key=f"rm_{product_id}")
        st.sidebar.caption("Material Group Overrides")
        for mg in ["Aluminum", "Steel", "Plastic", "Electronics"]:
            default_val = float(cfg["material_group_deltas"].get(mg, cfg["material_default_delta"]))
            cfg["material_group_deltas"][mg] = st.sidebar.number_input(
                f"{product_id} - {mg}", min_value=-0.5, max_value=1.0, value=default_val, step=0.01, key=f"{product_id}_{mg}"
            )
        st.session_state["scenario_configs"][product_id] = cfg
        st.sidebar.markdown("---")


def to_excel_bytes(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        detail_df.to_excel(writer, index=False, sheet_name="Detail")
    return output.getvalue()


init_state()

st.title("🌲 Multi-Product Risk Tree Dashboard")
st.caption("여러 제품 선택, 제품별 트리 구조, 리스크 상황 조회, 제품별 시나리오 설정")

tab1, tab2, tab3 = st.tabs(["Upload", "Dashboard", "Detail / Export"])

with tab1:
    uploaded = st.file_uploader("BOM 업로드 (CSV/XLSX)", type=["csv", "xlsx"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("샘플 BOM 불러오기", use_container_width=True):
            st.session_state["bom_df"] = normalize_df(sample_bom())
            st.session_state["upload_filename"] = "sample_bom"
            st.success("샘플 데이터를 불러왔습니다.")
            st.rerun()
    with c2:
        template = sample_bom()[ALL_COLUMNS].head(0)
        st.download_button(
            "빈 템플릿 다운로드",
            data=template.to_csv(index=False).encode("utf-8-sig"),
            file_name="multi_product_bom_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.session_state["bom_df"] = normalize_df(raw)
            st.session_state["upload_filename"] = uploaded.name
            st.success(f"업로드 완료: {uploaded.name}")
        except Exception as e:
            st.error(f"파일 처리 오류: {e}")

    if not st.session_state["bom_df"].empty:
        st.dataframe(st.session_state["bom_df"], use_container_width=True, hide_index=True)

bom_df = st.session_state["bom_df"].copy()
if not bom_df.empty:
    scenario_sidebar(bom_df)
    detail_df, summary_df = run_scenario(bom_df, st.session_state["selected_product_ids"])
else:
    detail_df, summary_df = pd.DataFrame(), pd.DataFrame()

with tab2:
    if bom_df.empty:
        st.info("먼저 Upload 탭에서 BOM을 올리거나 샘플 데이터를 불러와줘.")
    elif not st.session_state["selected_product_ids"]:
        st.info("왼쪽 사이드바에서 제품을 선택해줘.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total_base = summary_df["total_base_cost"].sum() if not summary_df.empty else 0
        total_scenario = summary_df["total_scenario_cost"].sum() if not summary_df.empty else 0
        total_delta = ((total_scenario - total_base) / total_base * 100) if total_base > 0 else 0
        c1.metric("선택 제품 수", len(summary_df))
        c2.metric("총 Base Cost", f"{total_base:,.0f}")
        c3.metric("총 Scenario Cost", f"{total_scenario:,.0f}")
        c4.metric("전체 원가 변화", f"{total_delta:.1f}%")

        st.markdown("### Product Summary")
        if not summary_df.empty:
            show_summary = summary_df.copy()
            show_summary["delta_pct"] = (show_summary["delta_pct"] * 100).round(1).astype(str) + "%"
            st.dataframe(show_summary, use_container_width=True, hide_index=True)

        st.markdown("### Product Trees")
        cols = st.columns(len(st.session_state["selected_product_ids"]))
        for i, product_id in enumerate(st.session_state["selected_product_ids"]):
            with cols[i]:
                st.markdown(f"#### {product_id}")
                render_tree(detail_df[detail_df["product_id"] == product_id], product_id)

with tab3:
    if detail_df.empty:
        st.info("결과 데이터가 없습니다.")
    else:
        left, right = st.columns([1.1, 1.9])
        with left:
            st.markdown("### Selected Part Detail")
            selected_part_no = st.session_state.get("selected_part_no")
            if selected_part_no:
                match = detail_df[detail_df["part_no"] == selected_part_no]
                if not match.empty:
                    row = match.iloc[0]
                    st.write(f"**Product**: {row['product_id']}")
                    st.write(f"**Part**: {row['part_name']} ({row['part_no']})")
                    st.write(f"**Material**: {row.get('material','')}")
                    st.write(f"**Supplier/Country**: {row.get('supplier','')} / {row.get('country','')}")
                    st.write(f"**Base Cost**: {row.get('base_cost',0):,.0f}")
                    st.write(f"**Scenario Cost**: {row.get('scenario_cost',0):,.0f}")
                    st.write(f"**Cost Delta**: {row.get('delta_pct',0)*100:.1f}%")
                    st.write(f"**Risk Score**: {row.get('risk_score',0):.2f}")
                    st.write(f"**Risk Status**: {row.get('risk_status','')}")
            else:
                st.info("트리에서 부품을 선택해줘.")

            st.markdown("### High Risk Parts")
            high_risk = detail_df[detail_df["risk_status"] == "High"][
                ["product_id", "part_no", "part_name", "supplier", "country", "scenario_cost", "delta_pct", "risk_score"]
            ]
            if high_risk.empty:
                st.success("현재 선택 제품에는 High Risk 부품이 없어.")
            else:
                tmp = high_risk.copy()
                tmp["delta_pct"] = (tmp["delta_pct"] * 100).round(1)
                st.dataframe(tmp, use_container_width=True, hide_index=True)

        with right:
            st.markdown("### Detail Table")
            tmp = detail_df.copy()
            tmp["delta_pct"] = (tmp["delta_pct"] * 100).round(1)
            st.dataframe(
                tmp[[
                    "product_id", "part_no", "parent_part_no", "part_name", "material_group",
                    "supplier", "country", "base_cost", "scenario_cost", "delta_pct", "risk_score", "risk_status"
                ]],
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = detail_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Detail CSV 다운로드", csv_bytes, "multi_product_risk_detail.csv", "text/csv", use_container_width=True)
            excel_bytes = to_excel_bytes(summary_df, detail_df)
            st.download_button(
                "Summary + Detail Excel 다운로드",
                excel_bytes,
                "multi_product_risk_dashboard.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
