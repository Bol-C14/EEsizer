from __future__ import annotations

from typing import Any

import pandas as pd

from ..contracts.enums import SimKind
from ..contracts.errors import MetricError, ValidationError
from ..io.ngspice_wrdata import load_wrdata_table
from ..sim.artifacts import RawSimData
from .registry import MetricImplSpec


def _missing(reason: str, debug: dict[str, Any] | None = None) -> tuple[None, dict]:
    details: dict[str, Any] = {"status": "missing", "reason": reason}
    if debug:
        details["debug"] = debug
    return None, details


def _pick_column(df: pd.DataFrame, target: str) -> pd.Series | None:
    target_lower = target.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return df[col]
    return None


def compute_power_w(raw: RawSimData, spec: MetricImplSpec) -> tuple[float | None, dict]:
    if raw.kind != SimKind.dc:
        raise ValidationError("Power metric requires SimKind.dc data")

    dc_path = raw.outputs.get("dc_csv")
    if dc_path is None:
        return _missing("missing_output:dc_csv")

    expected_cols = list(raw.outputs_meta.get("dc_csv", ())) or None
    try:
        _, df = load_wrdata_table(dc_path, expected_columns=expected_cols)
    except MetricError as exc:
        return _missing(f"dc_load_failed:{exc}")

    current_probe = str(spec.params.get("current_probe", "i(VDD)"))
    vdd_node = spec.params.get("vdd_node", "vdd")
    vdd_value_param = spec.params.get("vdd_value")

    current_col = _pick_column(df, current_probe)
    if current_col is None:
        return _missing(f"missing_probe:{current_probe}")
    current_series = current_col.dropna()
    if current_series.empty:
        return _missing("empty_current_probe")
    i_vdd = float(current_series.iloc[-1])

    vdd_value = None
    if vdd_value_param is not None:
        try:
            vdd_value = float(vdd_value_param)
        except (TypeError, ValueError):
            return _missing("invalid_vdd_value")

    vdd_expr = f"v({vdd_node})"
    vdd_col = _pick_column(df, vdd_expr)
    if vdd_col is not None:
        vdd_series = vdd_col.dropna()
        if not vdd_series.empty:
            vdd_value = float(vdd_series.iloc[-1])

    if vdd_value is None:
        return _missing(f"missing_probe:{vdd_expr}")

    power = abs(i_vdd) * abs(vdd_value)
    details = {"status": "ok", "debug": {"i_vdd": i_vdd, "vdd": vdd_value}}
    return float(power), details
