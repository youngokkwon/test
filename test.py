import io
import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Decision Risk Lab - BOM Page", layout="wide")


# =========================
# Config
# =========================
REQUIRED_COLUMNS = [
    "part_no",
    "parent_part_no",
    "part_name",
    "qty",
    "material",
    "weight",
    "unit_cost",
]

OPTIONAL_COLUMNS = [
    "supplier",
    "country",
    "process",
    "currency",
    "material_group",
    "ef_mapping_key",
    "ef_value",
    "ef_unit",
    "commodity_mapping_key",
    "commodity_price",
    "commodity_unit",
    "fx_exposure",
    "energy_intensity",
    "volatility_beta",
    "supply_risk_score",
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

LINKING_OPTIONS = [
    "배출계수",
    "원자재단가",
    "배출계수 + 원자재단가",
    "배출계수 + 원자재단가(변동성반영)",
]


# =========================
# Helpers
# =========================
def init_session_state():
    defaults = {
        "bom_df": pd.DataFrame(columns=ALL_COLUMNS + ["node_id", "level", "node_type", "status"]),
        "selected_part_no": None,
        "linking_mode": LINKING_OPTIONS[0],
        "validation_result": {},
        "mapping_preview": {},
        "upload_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def generate_template_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "part_no": "P-1000",
                "parent_part_no": "",
                "part_name": "Main Product",
                "qty": 1,
                "material": "",
                "weight": 0.0,
                "unit_cost": 0.0,
                "supplier": "",
                "country": "",
                "process": "",
                "currency": "KRW",
                "material_group": "",
                "ef_mapping_key": "",
                "ef_value": "",
                "ef_unit": "kgCO2e/kg",
                "commodity_mapping_key": "",
                "commodity_price": "",
                "commodity_unit": "USD/kg",
                "fx_exposure": "",
                "energy_intensity": "",
                "volatility_beta": "",
                "supply_risk_score": "",
            },
            {
                "part_no": "M-1100",
                "parent_part_no": "P-1000",
                "part_name": "Power Module",
                "qty": 1,
                "material": "",
                "weight": 0.0,
                "unit_cost": 0.0,
                "supplier": "",
                "country": "KR",
                "process": "",
                "currency": "KRW",
                "material_group": "",
                "ef_mapping_key": "",
                "ef_value": "",
                "ef_unit": "kgCO2e/kg",
                "commodity_mapping_key": "",
                "commodity_price": "",
                "commodity_unit": "USD/kg",
                "fx_exposure": "",
                "energy_intensity": "",
                "volatility_beta": "",
                "supply_risk_score": "",
            },
            {
                "part_no": "C-1110",
                "parent_part_no": "M-1100",
                "part_name": "Housing",
                "qty": 1,
                "material": "Aluminum ADC12",
                "weight": 1.2,
                "unit_cost": 4500,
                "supplier": "ABC Metal",
                "country": "KR",
                "process": "Die Casting",
                "currency": "KRW",
                "material_group": "Aluminum",
                "ef_mapping_key": "aluminum_adc12",
                "ef_value": 8.7,
                "ef_unit": "kgCO2e/kg",
                "commodity_mapping_key": "aluminum",
                "commodity_price": 2.5,
                "commodity_unit": "USD/kg",
                "fx_exposure": "Y",
                "energy_intensity": 0.8,
                "volatility_beta": 0.7,
                "supply_risk_score": 0.2,
            },
            {
                "part_no": "C-1120",
                "parent_part_no": "M-1100",
                "part_name": "Bolt",
                "qty": 4,
                "material": "SWCH18A",
                "weight": 0.02,
                "unit_cost": 120,
                "supplier": "Fasten Co",
                "country": "CN",
                "process": "Cold Heading",
                "currency": "KRW",
                "material_group": "Steel",
                "ef_mapping_key": "steel_unalloyed",
                "ef_value": 2.1,
                "ef_unit": "kgCO2e/kg",
                "commodity_mapping_key": "steel",
                "commodity_price": 0.9,
                "commodity_unit": "USD/kg",
                "fx_exposure": "Y",
                "energy_intensity": 0.2,
                "volatility_beta": 0.5,
                "supply_risk_score": 0.4,
            },
        ]
    )


def normalize_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df = df[ALL_COLUMNS].copy()
    df["part_no"] = df["part_no"].astype(str).str.strip()
    df["parent_part_no"] = df["parent_part_no"].fillna("").astype(str).str.strip()
    df["part_name"] = df["part_name"].astype(str).str.strip()
    df["material"] = df["material"].astype(str).str.strip()

    numeric_cols = [
        "qty",
        "weight",
        "unit_cost",
        "ef_value",
        "commodity_price",
        "energy_intensity",
        "volatility_beta",
        "supply_risk_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["node_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    df["node_type"] = "part"
    df["level"] = 0
    df["status"] = "미검증"
    return df


def validate_bom(df: pd.DataFrame) -> Dict:
    result = {
        "missing_required_columns": [],
        "empty_part_no_rows": [],
        "duplicate_part_no": [],
        "missing_parent_rows": [],
        "self_parent_rows": [],
        "warnings": [],
        "is_valid": True,
    }

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            result["missing_required_columns"].append(col)

    if result["missing_required_columns"]:
        result["is_valid"] = False
        return result

    empty_part = df.index[df["part_no"].astype(str).str.strip() == ""].tolist()
    if empty_part:
        result["empty_part_no_rows"] = [i + 2 for i in empty_part]
        result["is_valid"] = False

    dupes = df[df["part_no"].duplicated(keep=False)]["part_no"].tolist()
    if dupes:
        result["duplicate_part_no"] = sorted(list(set(dupes)))
        result["is_valid"] = False

    existing_parts = set(df["part_no"].astype(str).tolist())
    for idx, row in df.iterrows():
        part_no = str(row["part_no"]).strip()
        parent = str(row["parent_part_no"]).strip()
        if parent and parent not in existing_parts:
            result["missing_parent_rows"].append(
                {"row": idx + 2, "part_no": part_no, "missing_parent": parent}
            )
        if parent and parent == part_no:
            result["self_parent_rows"].append({"row": idx + 2, "part_no": part_no})

    if result["missing_parent_rows"] or result["self_parent_rows"]:
        result["is_valid"] = False

    material_missing = df.index[df["material"].astype(str).str.strip() == ""].tolist()
    if material_missing:
        result["warnings"].append(f"재질(material) 누락 행 수: {len(material_missing)}")

    cost_missing = df["unit_cost"].isna().sum()
    if cost_missing > 0:
        result["warnings"].append(f"unit_cost 누락 행 수: {int(cost_missing)}")

    weight_missing = df["weight"].isna().sum()
    if weight_missing > 0:
        result["warnings"].append(f"weight 누락 행 수: {int(weight_missing)}")

    return result


def build_level_map(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parent_map = dict(zip(df["part_no"], df["parent_part_no"]))

    def get_level(part_no: str, visited=None) -> int:
        if visited is None:
            visited = set()
        if part_no in visited:
            return 0
        visited.add(part_no)

        parent = str(parent_map.get(part_no, "")).strip()
        if not parent:
            return 0
        if parent not in parent_map:
            return 0
        return 1 + get_level(parent, visited)

    df["level"] = df["part_no"].apply(get_level)
    return df


def build_status(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df = df.copy()

    def check_row(row) -> str:
        ef_ready = pd.notna(row.get("ef_value")) or str(row.get("ef_mapping_key", "")).strip() != ""
        price_ready = pd.notna(row.get("commodity_price")) or str(row.get("commodity_mapping_key", "")).strip() != ""
        vol_ready = (
            pd.notna(row.get("volatility_beta"))
            and pd.notna(row.get("energy_intensity"))
            and str(row.get("fx_exposure", "")).strip() != ""
        )

        if mode == "배출계수":
            return "연결완료" if ef_ready else "미연결"
        if mode == "원자재단가":
            return "연결완료" if price_ready else "미연결"
        if mode == "배출계수 + 원자재단가":
            return "연결완료" if (ef_ready and price_ready) else "부분연결"
        if mode == "배출계수 + 원자재단가(변동성반영)":
            if ef_ready and price_ready and vol_ready:
                return "연결완료"
            elif ef_ready or price_ready:
                return "부분연결"
            return "미연결"
        return "미연결"

    df["status"] = df.apply(check_row, axis=1)
    return df


def build_children_map(df: pd.DataFrame) -> Dict[str, List[str]]:
    children_map: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        parent = str(row["parent_part_no"]).strip()
        part = str(row["part_no"]).strip()
        children_map.setdefault(parent, []).append(part)
    return children_map


def get_row_by_part_no(df: pd.DataFrame, part_no: str) -> Optional[pd.Series]:
    match = df[df["part_no"] == part_no]
    if len(match) == 0:
        return None
    return match.iloc[0]


def get_root_nodes(df: pd.DataFrame) -> List[str]:
    return df[df["parent_part_no"].astype(str).str.strip() == ""]["part_no"].tolist()


def status_icon(status: str) -> str:
    if status == "연결완료":
        return "🟢"
    if status == "부분연결":
        return "🟠"
    if status == "미연결":
        return "🔴"
    return "⚪"


def render_tree(df: pd.DataFrame, selected_part_no: Optional[str], search_text: str = ""):
    children_map = build_children_map(df)

    def node_visible(part_no: str) -> bool:
        if not search_text:
            return True
        row = get_row_by_part_no(df, part_no)
        if row is None:
            return False
        s = f"{row['part_no']} {row['part_name']} {row.get('material', '')} {row.get('supplier', '')} {row.get('country', '')}".lower()
        return search_text.lower() in s

    def render_node(part_no: str):
        row = get_row_by_part_no(df, part_no)
        if row is None:
            return

        children = children_map.get(part_no, [])
        label = f"{status_icon(row['status'])} {row['part_no']} | {row['part_name']}"

        if not node_visible(part_no) and not any(descendant_visible(c, children_map) for c in children):
            return

        with st.expander(label, expanded=(selected_part_no == part_no)):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.caption(
                    f"Material: {row.get('material', '')} | Qty: {row.get('qty', '')} | Country: {row.get('country', '')}"
                )
            with c2:
                if st.button("선택", key=f"select_{part_no}"):
                    st.session_state["selected_part_no"] = part_no
                    st.rerun()

            for child in children:
                render_node(child)

    def descendant_visible(part_no: str, cmap: Dict[str, List[str]]) -> bool:
        if node_visible(part_no):
            return True
        for c in cmap.get(part_no, []):
            if descendant_visible(c, cmap):
                return True
        return False

    roots = get_root_nodes(df)
    if not roots:
        st.info("루트 노드가 없습니다. parent_part_no가 비어 있는 상위 노드가 필요합니다.")
        return

    for root in roots:
        render_node(root)


def preview_summary(df: pd.DataFrame, mode: str) -> Dict:
    total = len(df)
    connected = (df["status"] == "연결완료").sum()
    partial = (df["status"] == "부분연결").sum()
    disconnected = (df["status"] == "미연결").sum()

    ef_connected = (
        (df["ef_mapping_key"].astype(str).str.strip() != "") | (df["ef_value"].notna())
    ).sum()
    price_connected = (
        (df["commodity_mapping_key"].astype(str).str.strip() != "") | (df["commodity_price"].notna())
    ).sum()
    vol_ready = (
        df["volatility_beta"].notna() & df["energy_intensity"].notna() & (df["fx_exposure"].astype(str).str.strip() != "")
    ).sum()

    return {
        "total_nodes": int(total),
        "connected": int(connected),
        "partial": int(partial),
        "disconnected": int(disconnected),
        "ef_connected": int(ef_connected),
        "price_connected": int(price_connected),
        "vol_ready": int(vol_ready),
        "mode": mode,
        "connected_rate": round((connected / total * 100), 1) if total else 0.0,
    }


def auto_fill_simple_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    material_rules = {
        "aluminum": {"ef_mapping_key": "aluminum_generic", "commodity_mapping_key": "aluminum", "commodity_unit": "USD/kg"},
        "adc12": {"ef_mapping_key": "aluminum_adc12", "commodity_mapping_key": "aluminum", "commodity_unit": "USD/kg"},
        "copper": {"ef_mapping_key": "copper_generic", "commodity_mapping_key": "copper", "commodity_unit": "USD/kg"},
        "steel": {"ef_mapping_key": "steel_generic", "commodity_mapping_key": "steel", "commodity_unit": "USD/kg"},
        "swch": {"ef_mapping_key": "steel_unalloyed", "commodity_mapping_key": "steel", "commodity_unit": "USD/kg"},
        "resin": {"ef_mapping_key": "plastic_generic", "commodity_mapping_key": "petrochemical_proxy", "commodity_unit": "USD/kg"},
        "abs": {"ef_mapping_key": "plastic_abs", "commodity_mapping_key": "petrochemical_proxy", "commodity_unit": "USD/kg"},
        "pcb": {"ef_mapping_key": "pcb_generic", "commodity_mapping_key": "electronics_proxy", "commodity_unit": "USD/kg"},
    }

    for idx, row in df.iterrows():
        material = str(row.get("material", "")).lower()
        for keyword, mapped in material_rules.items():
            if keyword in material:
                if str(row.get("ef_mapping_key", "")).strip() == "":
                    df.at[idx, "ef_mapping_key"] = mapped["ef_mapping_key"]
                if str(row.get("commodity_mapping_key", "")).strip() == "":
                    df.at[idx, "commodity_mapping_key"] = mapped["commodity_mapping_key"]
                if str(row.get("commodity_unit", "")).strip() == "":
                    df.at[idx, "commodity_unit"] = mapped["commodity_unit"]
                break

    return df


def add_child_node(df: pd.DataFrame, parent_part_no: str) -> pd.DataFrame:
    df = df.copy()
    new_part_no = f"NEW-{str(uuid.uuid4())[:6].upper()}"
    new_row = {col: "" for col in ALL_COLUMNS}
    new_row.update(
        {
            "part_no": new_part_no,
            "parent_part_no": parent_part_no,
            "part_name": "New Part",
            "qty": 1,
            "weight": 0.0,
            "unit_cost": 0.0,
            "currency": "KRW",
            "node_id": str(uuid.uuid4())[:8],
            "node_type": "part",
            "level": 0,
            "status": "미검증",
        }
    )
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def delete_node_and_descendants(df: pd.DataFrame, part_no: str) -> pd.DataFrame:
    df = df.copy()
    children_map = build_children_map(df)
    to_delete = set()

    def collect(node: str):
        to_delete.add(node)
        for child in children_map.get(node, []):
            collect(child)

    collect(part_no)
    df = df[~df["part_no"].isin(to_delete)].copy()
    return df


def df_to_download_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="BOM")
    return output.getvalue()


# =========================
# UI
# =========================
init_session_state()

st.title("📘 Decision Risk Lab — BOM Page")
st.caption("BOM 업로드, 트리 구조 편집, 연결 데이터 옵션 선택, Preview까지 한 페이지에서 작업")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Upload", "Structure", "Data Linking", "Preview", "Save / Export"]
)

# =========================
# TAB 1: Upload
# =========================
with tab1:
    st.subheader("1) BOM 업로드")

    c1, c2 = st.columns([2, 1])

    with c1:
        uploaded = st.file_uploader(
            "Excel 또는 CSV 업로드",
            type=["xlsx", "csv"],
            help="표준 템플릿 또는 사내 BOM 파일 업로드",
        )

    with c2:
        template_df = generate_template_df()
        template_csv = template_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "템플릿 다운로드 (CSV)",
            data=template_csv,
            file_name="decision_risk_bom_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("#### 표준 컬럼")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded)
            else:
                raw_df = pd.read_excel(uploaded)

            norm_df = normalize_uploaded_df(raw_df)
            norm_df = build_level_map(norm_df)
            norm_df = build_status(norm_df, st.session_state["linking_mode"])
            validation = validate_bom(norm_df)

            st.session_state["bom_df"] = norm_df
            st.session_state["validation_result"] = validation
            st.session_state["upload_filename"] = uploaded.name

            st.success(f"업로드 완료: {uploaded.name}")
            st.dataframe(norm_df.head(20), use_container_width=True)

        except Exception as e:
            st.error(f"파일 처리 중 오류: {e}")

    if st.button("샘플 BOM 불러오기", use_container_width=False):
        sample_df = normalize_uploaded_df(generate_template_df())
        sample_df = build_level_map(sample_df)
        sample_df = build_status(sample_df, st.session_state["linking_mode"])
        st.session_state["bom_df"] = sample_df
        st.session_state["validation_result"] = validate_bom(sample_df)
        st.session_state["upload_filename"] = "sample_bom"
        st.success("샘플 BOM을 불러왔습니다.")
        st.rerun()

    st.markdown("---")
    st.subheader("검증 결과")
    validation = st.session_state["validation_result"]

    if not validation:
        st.info("아직 검증된 BOM이 없습니다.")
    else:
        if validation.get("is_valid", False):
            st.success("필수 구조 검증 통과")
        else:
            st.error("검증 오류가 있습니다.")

        if validation.get("missing_required_columns"):
            st.write("**누락 컬럼:**", validation["missing_required_columns"])
        if validation.get("empty_part_no_rows"):
            st.write("**part_no 누락 행:**", validation["empty_part_no_rows"])
        if validation.get("duplicate_part_no"):
            st.write("**중복 part_no:**", validation["duplicate_part_no"])
        if validation.get("missing_parent_rows"):
            st.write("**존재하지 않는 parent_part_no 참조:**")
            st.dataframe(pd.DataFrame(validation["missing_parent_rows"]), use_container_width=True)
        if validation.get("self_parent_rows"):
            st.write("**자기 자신을 부모로 참조한 행:**")
            st.dataframe(pd.DataFrame(validation["self_parent_rows"]), use_container_width=True)
        if validation.get("warnings"):
            for w in validation["warnings"]:
                st.warning(w)


# =========================
# TAB 2: Structure
# =========================
with tab2:
    st.subheader("2) BOM Structure & Editor")

    bom_df = st.session_state["bom_df"].copy()
    if bom_df.empty:
        st.info("먼저 Upload 탭에서 BOM을 업로드해줘.")
    else:
        left, center, right = st.columns([1.2, 1.6, 1.2])

        with left:
            st.markdown("### Tree Explorer")
            search_text = st.text_input("검색", placeholder="part_no, part_name, material, supplier...")
            render_tree(bom_df, st.session_state["selected_part_no"], search_text=search_text)

        with center:
            st.markdown("### Detail Editor")
            selected_part_no = st.session_state["selected_part_no"]

            if not selected_part_no and len(bom_df) > 0:
                selected_part_no = bom_df.iloc[0]["part_no"]
                st.session_state["selected_part_no"] = selected_part_no

            row = get_row_by_part_no(bom_df, selected_part_no) if selected_part_no else None

            if row is None:
                st.warning("선택된 노드가 없습니다.")
            else:
                idx = bom_df.index[bom_df["part_no"] == selected_part_no][0]

                with st.form("detail_editor_form"):
                    col_a, col_b = st.columns(2)

                    with col_a:
                        part_no = st.text_input("part_no", value=str(row["part_no"]))
                        parent_part_no = st.text_input("parent_part_no", value=str(row["parent_part_no"]))
                        part_name = st.text_input("part_name", value=str(row["part_name"]))
                        qty = st.number_input("qty", value=float(row["qty"]) if pd.notna(row["qty"]) else 1.0, step=1.0)
                        material = st.text_input("material", value=str(row["material"]))
                        weight = st.number_input(
                            "weight", value=float(row["weight"]) if pd.notna(row["weight"]) else 0.0, step=0.01
                        )
                        unit_cost = st.number_input(
                            "unit_cost", value=float(row["unit_cost"]) if pd.notna(row["unit_cost"]) else 0.0, step=1.0
                        )
                        currency = st.text_input("currency", value=str(row.get("currency", "")))

                    with col_b:
                        supplier = st.text_input("supplier", value=str(row.get("supplier", "")))
                        country = st.text_input("country", value=str(row.get("country", "")))
                        process = st.text_input("process", value=str(row.get("process", "")))
                        material_group = st.text_input("material_group", value=str(row.get("material_group", "")))
                        ef_mapping_key = st.text_input("ef_mapping_key", value=str(row.get("ef_mapping_key", "")))
                        ef_value = st.number_input(
                            "ef_value",
                            value=float(row["ef_value"]) if pd.notna(row["ef_value"]) else 0.0,
                            step=0.1,
                        )
                        commodity_mapping_key = st.text_input(
                            "commodity_mapping_key", value=str(row.get("commodity_mapping_key", ""))
                        )
                        commodity_price = st.number_input(
                            "commodity_price",
                            value=float(row["commodity_price"]) if pd.notna(row["commodity_price"]) else 0.0,
                            step=0.1,
                        )

                    col_c, col_d, col_e = st.columns(3)
                    with col_c:
                        fx_exposure = st.text_input("fx_exposure", value=str(row.get("fx_exposure", "")))
                    with col_d:
                        energy_intensity = st.number_input(
                            "energy_intensity",
                            value=float(row["energy_intensity"]) if pd.notna(row["energy_intensity"]) else 0.0,
                            step=0.1,
                        )
                    with col_e:
                        volatility_beta = st.number_input(
                            "volatility_beta",
                            value=float(row["volatility_beta"]) if pd.notna(row["volatility_beta"]) else 0.0,
                            step=0.1,
                        )

                    save_btn = st.form_submit_button("변경사항 저장", use_container_width=True)

                if save_btn:
                    bom_df.at[idx, "part_no"] = part_no.strip()
                    bom_df.at[idx, "parent_part_no"] = parent_part_no.strip()
                    bom_df.at[idx, "part_name"] = part_name.strip()
                    bom_df.at[idx, "qty"] = qty
                    bom_df.at[idx, "material"] = material.strip()
                    bom_df.at[idx, "weight"] = weight
                    bom_df.at[idx, "unit_cost"] = unit_cost
                    bom_df.at[idx, "currency"] = currency.strip()
                    bom_df.at[idx, "supplier"] = supplier.strip()
                    bom_df.at[idx, "country"] = country.strip()
                    bom_df.at[idx, "process"] = process.strip()
                    bom_df.at[idx, "material_group"] = material_group.strip()
                    bom_df.at[idx, "ef_mapping_key"] = ef_mapping_key.strip()
                    bom_df.at[idx, "ef_value"] = ef_value if ef_value != 0 else pd.NA
                    bom_df.at[idx, "commodity_mapping_key"] = commodity_mapping_key.strip()
                    bom_df.at[idx, "commodity_price"] = commodity_price if commodity_price != 0 else pd.NA
                    bom_df.at[idx, "fx_exposure"] = fx_exposure.strip()
                    bom_df.at[idx, "energy_intensity"] = energy_intensity if energy_intensity != 0 else pd.NA
                    bom_df.at[idx, "volatility_beta"] = volatility_beta if volatility_beta != 0 else pd.NA

                    bom_df = build_level_map(bom_df)
                    bom_df = build_status(bom_df, st.session_state["linking_mode"])
                    st.session_state["bom_df"] = bom_df
                    st.session_state["validation_result"] = validate_bom(bom_df)
                    st.success("저장 완료")
                    st.rerun()

                a1, a2, a3 = st.columns(3)
                with a1:
                    if st.button("하위 부품 추가", use_container_width=True):
                        updated = add_child_node(bom_df, selected_part_no)
                        updated = build_level_map(updated)
                        updated = build_status(updated, st.session_state["linking_mode"])
                        st.session_state["bom_df"] = updated
                        st.success("하위 부품 추가 완료")
                        st.rerun()

                with a2:
                    if st.button("행 복제", use_container_width=True):
                        copy_row = bom_df[bom_df["part_no"] == selected_part_no].copy()
                        if len(copy_row) > 0:
                            copy_row = copy_row.iloc[0].to_dict()
                            copy_row["part_no"] = f"{selected_part_no}_COPY"
                            copy_row["node_id"] = str(uuid.uuid4())[:8]
                            copy_row["status"] = "미검증"
                            updated = pd.concat([bom_df, pd.DataFrame([copy_row])], ignore_index=True)
                            updated = build_level_map(updated)
                            updated = build_status(updated, st.session_state["linking_mode"])
                            st.session_state["bom_df"] = updated
                            st.success("복제 완료")
                            st.rerun()

                with a3:
                    if st.button("노드 삭제", use_container_width=True):
                        updated = delete_node_and_descendants(bom_df, selected_part_no)
                        updated = build_level_map(updated)
                        updated = build_status(updated, st.session_state["linking_mode"])
                        st.session_state["bom_df"] = updated
                        st.session_state["selected_part_no"] = None
                        st.success("삭제 완료")
                        st.rerun()

                st.markdown("---")
                st.markdown("### 빠른 테이블 편집")
                editable_cols = [
                    "part_no",
                    "parent_part_no",
                    "part_name",
                    "qty",
                    "material",
                    "weight",
                    "unit_cost",
                    "supplier",
                    "country",
                    "currency",
                    "ef_mapping_key",
                    "ef_value",
                    "commodity_mapping_key",
                    "commodity_price",
                    "fx_exposure",
                    "energy_intensity",
                    "volatility_beta",
                ]
                edited_df = st.data_editor(
                    bom_df[editable_cols],
                    use_container_width=True,
                    num_rows="dynamic",
                    hide_index=True,
                    key="bom_table_editor",
                )
                if st.button("테이블 편집 반영", use_container_width=True):
                    for col in editable_cols:
                        st.session_state["bom_df"][col] = edited_df[col]
                    st.session_state["bom_df"] = build_level_map(st.session_state["bom_df"])
                    st.session_state["bom_df"] = build_status(
                        st.session_state["bom_df"], st.session_state["linking_mode"]
                    )
                    st.session_state["validation_result"] = validate_bom(st.session_state["bom_df"])
                    st.success("테이블 편집값 반영 완료")
                    st.rerun()

        with right:
            st.markdown("### Structure Summary")
            st.metric("총 노드 수", len(bom_df))
            st.metric("루트 노드 수", len(get_root_nodes(bom_df)))
            st.metric("최대 레벨", int(bom_df["level"].max()) if len(bom_df) > 0 else 0)

            st.markdown("### Status Count")
            status_count = bom_df["status"].value_counts(dropna=False)
            st.dataframe(
                status_count.rename_axis("status").reset_index(name="count"),
                use_container_width=True,
                hide_index=True,
            )


# =========================
# TAB 3: Data Linking
# =========================
with tab3:
    st.subheader("3) Data Linking")

    bom_df = st.session_state["bom_df"].copy()
    if bom_df.empty:
        st.info("먼저 BOM을 업로드해줘.")
    else:
        st.markdown("### 연결 옵션 선택")
        linking_mode = st.radio(
            "연결 모드",
            LINKING_OPTIONS,
            index=LINKING_OPTIONS.index(st.session_state["linking_mode"]),
            horizontal=False,
        )
        st.session_state["linking_mode"] = linking_mode

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("간단 자동 매핑 실행", use_container_width=True):
                updated = auto_fill_simple_mapping(bom_df)
                updated = build_level_map(updated)
                updated = build_status(updated, linking_mode)
                st.session_state["bom_df"] = updated
                st.success("자동 매핑 완료")
                st.rerun()

        with c2:
            if st.button("연결 상태 재계산", use_container_width=True):
                updated = build_status(bom_df, linking_mode)
                st.session_state["bom_df"] = updated
                st.success("재계산 완료")
                st.rerun()

        st.markdown("---")
        st.markdown("### 연결 상태 테이블")
        preview_cols = [
            "part_no",
            "part_name",
            "material",
            "ef_mapping_key",
            "ef_value",
            "commodity_mapping_key",
            "commodity_price",
            "fx_exposure",
            "energy_intensity",
            "volatility_beta",
            "status",
        ]
        st.dataframe(
            st.session_state["bom_df"][preview_cols],
            use_container_width=True,
            hide_index=True,
        )


# =========================
# TAB 4: Preview
# =========================
with tab4:
    st.subheader("4) Preview")

    bom_df = st.session_state["bom_df"].copy()
    if bom_df.empty:
        st.info("먼저 BOM을 업로드해줘.")
    else:
        bom_df = build_status(bom_df, st.session_state["linking_mode"])
        st.session_state["bom_df"] = bom_df
        summary = preview_summary(bom_df, st.session_state["linking_mode"])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("총 노드 수", summary["total_nodes"])
        r2.metric("연결 완료", summary["connected"])
        r3.metric("부분 연결", summary["partial"])
        r4.metric("미연결", summary["disconnected"])

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("연결 완료율", f"{summary['connected_rate']}%")
        r6.metric("EF 연결 가능", summary["ef_connected"])
        r7.metric("가격 연결 가능", summary["price_connected"])
        r8.metric("변동성 준비", summary["vol_ready"])

        st.markdown("---")
        st.markdown("### 미연결 / 부분연결 노드")
        issue_df = bom_df[bom_df["status"].isin(["미연결", "부분연결"])][
            [
                "part_no",
                "part_name",
                "material",
                "supplier",
                "country",
                "ef_mapping_key",
                "commodity_mapping_key",
                "fx_exposure",
                "energy_intensity",
                "volatility_beta",
                "status",
            ]
        ]
        if issue_df.empty:
            st.success("모든 노드가 현재 선택 모드 기준으로 연결 완료 상태야.")
        else:
            st.dataframe(issue_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 상위 원가 부품")
        if "unit_cost" in bom_df.columns:
            top_cost = bom_df.sort_values("unit_cost", ascending=False)[
                ["part_no", "part_name", "material", "unit_cost", "currency", "status"]
            ].head(10)
            st.dataframe(top_cost, use_container_width=True, hide_index=True)


# =========================
# TAB 5: Save / Export
# =========================
with tab5:
    st.subheader("5) Save / Export")

    bom_df = st.session_state["bom_df"].copy()
    if bom_df.empty:
        st.info("저장할 BOM이 없습니다.")
    else:
        st.markdown("### 현재 상태")
        st.write(f"- 업로드 파일: {st.session_state['upload_filename']}")
        st.write(f"- 연결 모드: {st.session_state['linking_mode']}")
        st.write(f"- 총 노드 수: {len(bom_df)}")

        export_df = bom_df.copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSV 다운로드",
            data=csv_data,
            file_name="decision_risk_bom_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

        try:
            xlsx_data = df_to_download_bytes(export_df)
            st.download_button(
                "Excel 다운로드",
                data=xlsx_data,
                file_name="decision_risk_bom_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Excel 다운로드 준비 중 오류: {e}")

        st.markdown("---")
        st.markdown("### Simulation Input Preview")
        sim_input_cols = [
            "part_no",
            "parent_part_no",
            "part_name",
            "qty",
            "material",
            "weight",
            "unit_cost",
            "currency",
            "ef_value",
            "commodity_price",
            "fx_exposure",
            "energy_intensity",
            "volatility_beta",
            "supply_risk_score",
            "status",
        ]
        available_cols = [c for c in sim_input_cols if c in export_df.columns]
        st.dataframe(export_df[available_cols], use_container_width=True, hide_index=True)

        st.success("이 데이터셋을 다음 시뮬레이션 페이지 입력용으로 바로 넘길 수 있는 상태야.")