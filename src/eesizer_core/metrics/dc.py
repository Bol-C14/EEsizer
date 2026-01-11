from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..contracts.enums import SimKind
from ..contracts.errors import MetricError, ValidationError
from ..sim.artifacts import RawSimData
from .registry import MetricSpec


def _load_dc_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MetricError(f"DC data file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    return df


def _pick_column(df: pd.DataFrame, target: str) -> pd.Series:
    target_lower = target.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return df[col]
    raise MetricError(f"Column '{target}' not found in DC data; available: {df.columns.tolist()}")


def compute_dc_vout_last(raw: RawSimData, spec: MetricSpec) -> float:
    if raw.kind != SimKind.dc:
        raise ValidationError(f"DC metric '{spec.name}' requires SimKind.dc data")
    dc_path = raw.outputs.get("dc_csv")
    if dc_path is None:
        raise ValidationError("RawSimData missing required output 'dc_csv'")

    df = _load_dc_dataframe(dc_path)
    node = str(spec.params.get("node", "out"))
    series = _pick_column(df, f"v({node})")
    if series.size == 0:
        raise MetricError("DC data empty for node")
    return float(series.dropna().iloc[-1])


def compute_dc_slope(raw: RawSimData, spec: MetricSpec) -> Tuple[float | None, dict]:
    """Approximate slope dV/dx using first/last samples; returns diagnostics on failure."""

    if raw.kind != SimKind.dc:
        raise ValidationError(f"DC metric '{spec.name}' requires SimKind.dc data")
    dc_path = raw.outputs.get("dc_csv")
    if dc_path is None:
        raise ValidationError("RawSimData missing required output 'dc_csv'")

    df = _load_dc_dataframe(dc_path)
    sweep_col = str(spec.params.get("sweep_col", "v(in)"))
    node = str(spec.params.get("node", "out"))

    try:
        sweep = _pick_column(df, sweep_col).to_numpy(dtype=float)
    except MetricError as exc:
        return None, {"reason": str(exc)}
    try:
        values = _pick_column(df, f"v({node})").to_numpy(dtype=float)
    except MetricError as exc:
        return None, {"reason": str(exc)}

    if sweep.size < 2 or values.size < 2:
        return None, {"reason": "insufficient samples"}

    dx = sweep[-1] - sweep[0]
    dy = values[-1] - values[0]
    if dx == 0:
        return None, {"reason": "zero sweep delta"}
    return float(dy / dx), {}
