from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..contracts.enums import SimKind
from ..contracts.errors import MetricError, ValidationError
from ..sim.artifacts import RawSimData
from .registry import MetricSpec


def _load_tran_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise MetricError(f"TRAN data file not found: {path}")
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    return df


def _pick_column(df: pd.DataFrame, target: str) -> pd.Series:
    target_lower = target.lower()
    for col in df.columns:
        if col.lower() == target_lower:
            return df[col]
    raise MetricError(f"Column '{target}' not found in TRAN data; available: {df.columns.tolist()}")


def _rise_time(time: np.ndarray, values: np.ndarray, low_frac: float = 0.1, high_frac: float = 0.9) -> float | None:
    if time.size < 2:
        return None
    v_min, v_max = float(np.min(values)), float(np.max(values))
    if v_max == v_min:
        return None
    low = v_min + (v_max - v_min) * low_frac
    high = v_min + (v_max - v_min) * high_frac

    def _cross(target: float) -> float | None:
        for i in range(1, time.size):
            if (values[i - 1] <= target <= values[i]) or (values[i - 1] >= target >= values[i]):
                if values[i] == values[i - 1]:
                    return float(time[i])
                frac = (target - values[i - 1]) / (values[i] - values[i - 1])
                return float(time[i - 1] + frac * (time[i] - time[i - 1]))
        return None

    t_low = _cross(low)
    t_high = _cross(high)
    if t_low is None or t_high is None:
        return None
    return max(0.0, t_high - t_low)


def compute_tran_rise_time(raw: RawSimData, spec: MetricSpec) -> Tuple[float | None, dict]:
    if raw.kind != SimKind.tran:
        raise ValidationError(f"TRAN metric '{spec.name}' requires SimKind.tran data")
    tran_path = raw.outputs.get("tran_csv")
    if tran_path is None:
        raise ValidationError("RawSimData missing required output 'tran_csv'")

    df = _load_tran_dataframe(tran_path)
    node = str(spec.params.get("node", "out"))
    time_series = _pick_column(df, "time").to_numpy(dtype=float)
    values = _pick_column(df, f"v({node})").to_numpy(dtype=float)

    rt = _rise_time(time_series, values)
    diagnostics = {}
    if rt is None:
        diagnostics["reason"] = "unable to find crossings"
    return rt, diagnostics
