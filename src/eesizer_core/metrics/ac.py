from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..contracts.errors import MetricError, ValidationError
from ..contracts.enums import SimKind
from ..sim.artifacts import RawSimData
from ..io.ngspice_wrdata import load_wrdata_table
from .registry import MetricImplSpec


def _interp_at(freq: np.ndarray, values: np.ndarray, target: float) -> float:
    if freq.size == 0:
        raise MetricError("AC data is empty")
    order = np.argsort(freq)
    freq = freq[order]
    values = values[order]
    if target <= freq[0]:
        return float(values[0])
    if target >= freq[-1]:
        return float(values[-1])
    return float(np.interp(target, freq, values))


def _first_crossing_from_above(freq: np.ndarray, values: np.ndarray, target: float) -> float:
    if freq.size < 2:
        raise MetricError("AC data has insufficient samples for crossing detection")
    for i in range(1, freq.size):
        prev = values[i - 1]
        curr = values[i]
        if prev >= target >= curr or prev <= target <= curr:
            if curr == prev:
                return float(freq[i])
            frac = (target - prev) / (curr - prev)
            return float(freq[i - 1] + frac * (freq[i] - freq[i - 1]))
    raise MetricError(f"No crossing found for target {target} dB")


def _extract_ac(raw: RawSimData, spec: MetricImplSpec) -> Tuple[np.ndarray, np.ndarray]:
    if raw.kind != SimKind.ac:
        raise ValidationError(f"AC metric '{spec.name}' requires SimKind.ac data")
    node = str(spec.params.get("node", "out"))
    expected_cols = list(raw.outputs_meta.get("ac_csv", ())) or ["frequency", f"real(v({node}))", f"imag(v({node}))"]
    ac_path = raw.outputs.get("ac_csv")
    if ac_path is None:
        raise ValidationError("RawSimData missing required output 'ac_csv'")
    _, df = load_wrdata_table(ac_path, expected_columns=expected_cols)

    def _pick_column(df: pd.DataFrame, target: str) -> pd.Series:
        target_lower = target.lower()
        for col in df.columns:
            if col.lower() == target_lower:
                return df[col]
        raise MetricError(f"Column '{target}' not found in AC data; available: {df.columns.tolist()}")

    freq = _pick_column(df, "frequency").to_numpy(dtype=float)
    real = _pick_column(df, f"real(v({node}))").to_numpy(dtype=float)
    imag = _pick_column(df, f"imag(v({node}))").to_numpy(dtype=float)
    mag = np.sqrt(real**2 + imag**2)
    mag = np.maximum(mag, 1e-30)
    mag_db_col = 20 * np.log10(mag)
    return freq, mag_db_col


def compute_ac_mag_db_at(raw: RawSimData, spec: MetricImplSpec) -> float:
    target_hz = spec.params.get("target_hz")
    if target_hz is None:
        raise ValidationError("ac_mag_db_at requires 'target_hz' in params")
    freq, mag_db = _extract_ac(raw, spec)
    return _interp_at(freq, mag_db, float(target_hz))


def compute_unity_gain_freq(raw: RawSimData, spec: MetricImplSpec) -> float | tuple[float | None, dict]:
    freq, mag_db = _extract_ac(raw, spec)
    target = float(spec.params.get("target_db", 0.0))
    try:
        return _first_crossing_from_above(freq, mag_db, target)
    except MetricError as exc:
        return None, {"reason": str(exc)}
