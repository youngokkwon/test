from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_COLUMNS = {
    "part_no",
    "part_name",
    "material",
    "material_group",
    "supplier",
    "country",
    "qty",
    "weight",
    "unit_cost",
    "currency",
    "commodity_mapping_key",
    "commodity_price",
    "ef_value",
    "fx_exposure",
    "energy_intensity",
    "volatility_beta",
    "supply_risk_score",
}


@dataclass
class ScenarioConfig:
    """Scenario input for BOM cost simulation.

    Percent fields are passed as decimal ratios.
    Example:
        fx_delta = 0.10  -> +10%
        energy_delta = -0.05 -> -5%
    """

    scenario_name: str = "Base Scenario"
    fx_delta: float = 0.0
    energy_delta: float = 0.0
    logistics_delta: float = 0.0
    risk_multiplier: float = 0.0
    default_material_delta: float = 0.0
    material_deltas: Optional[Dict[str, float]] = None
    country_risk_deltas: Optional[Dict[str, float]] = None
    supplier_risk_deltas: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScenarioEngineError(Exception):
    """Raised when scenario input is invalid."""


class ScenarioEngine:
    """Run scenario-based BOM cost simulation.

    Expected BOM columns (flexible, not all mandatory):
        - part_no
        - part_name
        - material
        - material_group
        - supplier
        - country
        - qty
        - weight
        - unit_cost
        - commodity_price
        - fx_exposure
        - energy_intensity
        - volatility_beta
        - supply_risk_score

    Notes:
        1) If commodity_price exists, material base cost is estimated as qty * weight * commodity_price.
        2) If commodity_price is missing, unit_cost is used as fallback base material cost.
        3) Energy cost is approximated from base total cost * energy_intensity.
        4) FX effect is applied only when fx_exposure indicates yes.
        5) Risk premium uses supply_risk_score plus optional country/supplier overlays.
    """

    YES_VALUES = {"y", "yes", "true", "1", "fx", "exposed"}

    def __init__(self, bom_df: pd.DataFrame):
        self.original_df = bom_df.copy()
        self.df = self._prepare_df(bom_df)

    @staticmethod
    def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ScenarioEngineError("BOM dataframe is empty.")

        work = df.copy()
        for col in DEFAULT_COLUMNS:
            if col not in work.columns:
                work[col] = pd.NA

        text_cols = [
            "part_no",
            "part_name",
            "material",
            "material_group",
            "supplier",
            "country",
            "currency",
            "commodity_mapping_key",
            "fx_exposure",
        ]
        for col in text_cols:
            work[col] = work[col].fillna("").astype(str).str.strip()

        numeric_cols = [
            "qty",
            "weight",
            "unit_cost",
            "commodity_price",
            "ef_value",
            "energy_intensity",
            "volatility_beta",
            "supply_risk_score",
        ]
        for col in numeric_cols:
            work[col] = pd.to_numeric(work[col], errors="coerce")

        work["qty"] = work["qty"].fillna(1.0)
        work["weight"] = work["weight"].fillna(0.0)
        work["unit_cost"] = work["unit_cost"].fillna(0.0)
        work["commodity_price"] = work["commodity_price"].fillna(pd.NA)
        work["energy_intensity"] = work["energy_intensity"].fillna(0.0)
        work["volatility_beta"] = work["volatility_beta"].fillna(0.0)
        work["supply_risk_score"] = work["supply_risk_score"].fillna(0.0)

        if "part_no" in work.columns:
            blank_mask = work["part_no"].eq("")
            if blank_mask.any():
                work.loc[blank_mask, "part_no"] = [f"ROW-{i}" for i in work.index[blank_mask]]

        return work

    @classmethod
    def from_file(cls, filepath: str, sheet_name: Optional[str] = None) -> "ScenarioEngine":
        if filepath.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
        return cls(df)

    def run(self, config: ScenarioConfig) -> pd.DataFrame:
        df = self.df.copy()

        df["scenario_name"] = config.scenario_name
        df["base_material_cost"] = df.apply(self._calc_base_material_cost, axis=1)
        df["base_energy_cost"] = df.apply(self._calc_base_energy_cost, axis=1)
        df["base_other_cost"] = df.apply(self._calc_base_other_cost, axis=1)
        df["base_total_cost"] = (
            df["base_material_cost"] + df["base_energy_cost"] + df["base_other_cost"]
        )

        df["material_delta"] = df.apply(lambda row: self._resolve_material_delta(row, config), axis=1)
        df["country_risk_delta"] = df.apply(lambda row: self._resolve_country_risk_delta(row, config), axis=1)
        df["supplier_risk_delta"] = df.apply(lambda row: self._resolve_supplier_risk_delta(row, config), axis=1)
        df["effective_risk_score"] = (
            df["supply_risk_score"] + df["country_risk_delta"] + df["supplier_risk_delta"]
        ).clip(lower=0.0)

        df["scenario_material_cost"] = df["base_material_cost"] * (1.0 + df["material_delta"])
        df["scenario_energy_cost"] = df["base_energy_cost"] * (1.0 + config.energy_delta)
        df["scenario_other_cost"] = df["base_other_cost"] * (1.0 + config.logistics_delta)

        df["subtotal_before_fx"] = (
            df["scenario_material_cost"] + df["scenario_energy_cost"] + df["scenario_other_cost"]
        )
        df["fx_multiplier"] = df["fx_exposure"].apply(
            lambda x: 1.0 + config.fx_delta if self._is_fx_exposed(x) else 1.0
        )
        df["subtotal_after_fx"] = df["subtotal_before_fx"] * df["fx_multiplier"]

        df["volatility_premium"] = (
            df["subtotal_after_fx"] * df["volatility_beta"].fillna(0.0).clip(lower=0.0)
            * abs(df["material_delta"])
        )
        df["risk_premium"] = (
            df["subtotal_after_fx"] * df["effective_risk_score"] * config.risk_multiplier
        )

        df["scenario_total_cost"] = (
            df["subtotal_after_fx"] + df["volatility_premium"] + df["risk_premium"]
        )
        df["cost_delta_abs"] = df["scenario_total_cost"] - df["base_total_cost"]
        df["cost_delta_pct"] = df.apply(self._safe_delta_pct, axis=1)
        df["top_driver"] = df.apply(self._identify_top_driver, axis=1, config=config)

        ordered = self._ordered_output(df)
        return ordered

    def summarize(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        if result_df is None or len(result_df) == 0:
            return {
                "total_parts": 0,
                "total_base_cost": 0.0,
                "total_scenario_cost": 0.0,
                "delta_abs": 0.0,
                "delta_pct": 0.0,
                "top_driver": None,
            }

        total_base = float(result_df["base_total_cost"].sum())
        total_scenario = float(result_df["scenario_total_cost"].sum())
        delta_abs = total_scenario - total_base
        delta_pct = (delta_abs / total_base) if total_base else 0.0

        top_driver = None
        if "top_driver" in result_df.columns and not result_df["top_driver"].empty:
            top_driver = result_df.groupby("top_driver")["cost_delta_abs"].sum().sort_values(ascending=False)
            top_driver = top_driver.index[0] if len(top_driver) > 0 else None

        return {
            "total_parts": int(len(result_df)),
            "total_base_cost": round(total_base, 4),
            "total_scenario_cost": round(total_scenario, 4),
            "delta_abs": round(delta_abs, 4),
            "delta_pct": round(delta_pct, 6),
            "top_driver": top_driver,
        }

    def compare_scenarios(self, scenarios: Iterable[ScenarioConfig]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        summary_rows: List[Dict[str, Any]] = []
        detail_map: Dict[str, pd.DataFrame] = {}

        for config in scenarios:
            result_df = self.run(config)
            detail_map[config.scenario_name] = result_df
            summary = self.summarize(result_df)
            summary_rows.append(
                {
                    "scenario_name": config.scenario_name,
                    "total_parts": summary["total_parts"],
                    "total_base_cost": summary["total_base_cost"],
                    "total_scenario_cost": summary["total_scenario_cost"],
                    "delta_abs": summary["delta_abs"],
                    "delta_pct": summary["delta_pct"],
                    "top_driver": summary["top_driver"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values("delta_abs", ascending=False).reset_index(drop=True)
        return summary_df, detail_map

    @staticmethod
    def _calc_base_material_cost(row: pd.Series) -> float:
        qty = float(row.get("qty", 1.0) or 1.0)
        weight = float(row.get("weight", 0.0) or 0.0)
        unit_cost = float(row.get("unit_cost", 0.0) or 0.0)
        commodity_price = row.get("commodity_price", pd.NA)

        if pd.notna(commodity_price) and weight > 0:
            return qty * weight * float(commodity_price)
        return qty * unit_cost

    @staticmethod
    def _calc_base_energy_cost(row: pd.Series) -> float:
        qty = float(row.get("qty", 1.0) or 1.0)
        unit_cost = float(row.get("unit_cost", 0.0) or 0.0)
        energy_intensity = float(row.get("energy_intensity", 0.0) or 0.0)
        base_total_proxy = qty * unit_cost
        return base_total_proxy * energy_intensity

    @staticmethod
    def _calc_base_other_cost(row: pd.Series) -> float:
        qty = float(row.get("qty", 1.0) or 1.0)
        unit_cost = float(row.get("unit_cost", 0.0) or 0.0)
        energy_intensity = float(row.get("energy_intensity", 0.0) or 0.0)
        material_estimate = ScenarioEngine._calc_base_material_cost(row)
        energy_estimate = ScenarioEngine._calc_base_energy_cost(row)
        total_proxy = qty * unit_cost
        other = total_proxy - material_estimate - energy_estimate
        return max(other, 0.0)

    @staticmethod
    def _is_fx_exposed(value: Any) -> bool:
        return str(value).strip().lower() in ScenarioEngine.YES_VALUES

    @staticmethod
    def _match_delta(mapping: Optional[Dict[str, float]], keys: List[str]) -> Optional[float]:
        if not mapping:
            return None
        lowered = {str(k).strip().lower(): float(v) for k, v in mapping.items()}
        for key in keys:
            k = str(key).strip().lower()
            if k and k in lowered:
                return lowered[k]
        return None

    def _resolve_material_delta(self, row: pd.Series, config: ScenarioConfig) -> float:
        keys = [
            row.get("commodity_mapping_key", ""),
            row.get("material_group", ""),
            row.get("material", ""),
        ]
        matched = self._match_delta(config.material_deltas, keys)
        return float(matched) if matched is not None else float(config.default_material_delta)

    def _resolve_country_risk_delta(self, row: pd.Series, config: ScenarioConfig) -> float:
        matched = self._match_delta(config.country_risk_deltas, [row.get("country", "")])
        return float(matched) if matched is not None else 0.0

    def _resolve_supplier_risk_delta(self, row: pd.Series, config: ScenarioConfig) -> float:
        matched = self._match_delta(config.supplier_risk_deltas, [row.get("supplier", "")])
        return float(matched) if matched is not None else 0.0

    @staticmethod
    def _safe_delta_pct(row: pd.Series) -> float:
        base = float(row.get("base_total_cost", 0.0) or 0.0)
        delta = float(row.get("cost_delta_abs", 0.0) or 0.0)
        return (delta / base) if base else 0.0

    @staticmethod
    def _identify_top_driver(row: pd.Series, config: ScenarioConfig) -> str:
        impacts = {
            "Material": float(row.get("scenario_material_cost", 0.0) - row.get("base_material_cost", 0.0)),
            "Energy": float(row.get("scenario_energy_cost", 0.0) - row.get("base_energy_cost", 0.0)),
            "FX": float(row.get("subtotal_after_fx", 0.0) - row.get("subtotal_before_fx", 0.0)),
            "Risk": float(row.get("risk_premium", 0.0)),
            "Volatility": float(row.get("volatility_premium", 0.0)),
            "Logistics": float(row.get("scenario_other_cost", 0.0) - row.get("base_other_cost", 0.0)),
        }
        return max(impacts.items(), key=lambda x: abs(x[1]))[0]

    @staticmethod
    def _ordered_output(df: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "scenario_name",
            "part_no",
            "part_name",
            "material",
            "material_group",
            "supplier",
            "country",
            "qty",
            "weight",
            "unit_cost",
            "currency",
            "commodity_mapping_key",
            "commodity_price",
            "fx_exposure",
            "energy_intensity",
            "volatility_beta",
            "supply_risk_score",
            "material_delta",
            "country_risk_delta",
            "supplier_risk_delta",
            "effective_risk_score",
            "base_material_cost",
            "base_energy_cost",
            "base_other_cost",
            "base_total_cost",
            "scenario_material_cost",
            "scenario_energy_cost",
            "scenario_other_cost",
            "subtotal_before_fx",
            "fx_multiplier",
            "subtotal_after_fx",
            "volatility_premium",
            "risk_premium",
            "scenario_total_cost",
            "cost_delta_abs",
            "cost_delta_pct",
            "top_driver",
        ]
        remaining = [c for c in df.columns if c not in preferred]
        return df[preferred + remaining].copy()


def run_scenario(
    bom_df: pd.DataFrame,
    scenario_name: str = "Scenario",
    fx_delta: float = 0.0,
    energy_delta: float = 0.0,
    logistics_delta: float = 0.0,
    risk_multiplier: float = 0.0,
    default_material_delta: float = 0.0,
    material_deltas: Optional[Dict[str, float]] = None,
    country_risk_deltas: Optional[Dict[str, float]] = None,
    supplier_risk_deltas: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    engine = ScenarioEngine(bom_df)
    config = ScenarioConfig(
        scenario_name=scenario_name,
        fx_delta=fx_delta,
        energy_delta=energy_delta,
        logistics_delta=logistics_delta,
        risk_multiplier=risk_multiplier,
        default_material_delta=default_material_delta,
        material_deltas=material_deltas,
        country_risk_deltas=country_risk_deltas,
        supplier_risk_deltas=supplier_risk_deltas,
    )
    result_df = engine.run(config)
    summary = engine.summarize(result_df)
    return result_df, summary


if __name__ == "__main__":
    sample = pd.DataFrame(
        [
            {
                "part_no": "C-1110",
                "part_name": "Housing",
                "material": "Aluminum ADC12",
                "material_group": "Aluminum",
                "supplier": "ABC Metal",
                "country": "KR",
                "qty": 1,
                "weight": 1.2,
                "unit_cost": 4500,
                "currency": "KRW",
                "commodity_mapping_key": "aluminum",
                "commodity_price": 2.5,
                "fx_exposure": "Y",
                "energy_intensity": 0.08,
                "volatility_beta": 0.7,
                "supply_risk_score": 0.2,
            },
            {
                "part_no": "C-1120",
                "part_name": "Bolt",
                "material": "SWCH18A",
                "material_group": "Steel",
                "supplier": "Fasten Co",
                "country": "CN",
                "qty": 4,
                "weight": 0.02,
                "unit_cost": 120,
                "currency": "KRW",
                "commodity_mapping_key": "steel",
                "commodity_price": 0.9,
                "fx_exposure": "Y",
                "energy_intensity": 0.03,
                "volatility_beta": 0.5,
                "supply_risk_score": 0.4,
            },
        ]
    )

    scenario = ScenarioConfig(
        scenario_name="Stress Case",
        fx_delta=0.1,
        energy_delta=0.15,
        logistics_delta=0.05,
        risk_multiplier=0.2,
        default_material_delta=0.03,
        material_deltas={"aluminum": 0.12, "steel": 0.06},
        country_risk_deltas={"cn": 0.15},
    )

    engine = ScenarioEngine(sample)
    result = engine.run(scenario)
    summary = engine.summarize(result)

    print("=== SUMMARY ===")
    print(summary)
    print("\n=== DETAIL ===")
    print(result[["part_no", "base_total_cost", "scenario_total_cost", "cost_delta_pct", "top_driver"]])
