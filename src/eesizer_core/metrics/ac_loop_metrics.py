from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..contracts.enums import SimKind
from ..contracts.errors import MetricError, ValidationError
from ..io.ngspice_wrdata import load_wrdata_table
from ..sim.artifacts import RawSimData
from .registry import MetricImplSpec


_MAG_EPS = 1e-30


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


def _complex_for_expr(df: pd.DataFrame, expr: str) -> tuple[np.ndarray | None, str | None]:
    real_col = _pick_column(df, f"real({expr})")
    imag_col = _pick_column(df, f"imag({expr})")
    if real_col is None or imag_col is None:
        return None, f"missing_probe:{expr}"
    real = real_col.to_numpy(dtype=float)
    imag = imag_col.to_numpy(dtype=float)
    if real.size != imag.size:
        return None, f"mismatched_probe:{expr}"
    return real + 1j * imag, None


def _expr_from_node(node: str | None) -> str | None:
    if node is None:
        return None
    node = str(node).strip()
    if not node:
        return None
    return f"v({node})"


def _load_ac_table(raw: RawSimData) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    if raw.kind != SimKind.ac:
        raise ValidationError("AC loop metrics require SimKind.ac data")
    ac_path = raw.outputs.get("ac_csv")
    if ac_path is None:
        return None, {"reason": "missing_output:ac_csv"}
    expected_cols = list(raw.outputs_meta.get("ac_csv", ())) or None
    try:
        _, df = load_wrdata_table(ac_path, expected_columns=expected_cols)
    except MetricError as exc:
        return None, {"reason": f"ac_load_failed:{exc}"}
    return df, None


def _prepare_transfer(raw: RawSimData, spec: MetricImplSpec) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    df, err = _load_ac_table(raw)
    if df is None:
        return np.array([]), np.array([]), err or {"reason": "missing_ac"}

    freq_col = _pick_column(df, "frequency")
    if freq_col is None:
        return np.array([]), np.array([]), {"reason": "missing_frequency"}
    freq = freq_col.to_numpy(dtype=float)
    if freq.size < 2:
        return np.array([]), np.array([]), {"reason": "insufficient_samples"}
    if np.any(freq <= 0):
        return np.array([]), np.array([]), {"reason": "nonpositive_frequency"}

    output_expr = spec.params.get("output_expr") or _expr_from_node(spec.params.get("output_node", "vout"))
    if not isinstance(output_expr, str):
        return np.array([]), np.array([]), {"reason": "missing_output_expr"}
    vout, err = _complex_for_expr(df, output_expr)
    if vout is None:
        return np.array([]), np.array([]), {"reason": err or "missing_output_probe"}

    input_pos_expr = _expr_from_node(spec.params.get("input_pos"))
    input_neg_expr = _expr_from_node(spec.params.get("input_neg"))
    input_single_expr = _expr_from_node(spec.params.get("input_node"))

    if input_pos_expr and input_neg_expr:
        vinp, err = _complex_for_expr(df, input_pos_expr)
        if vinp is None:
            return np.array([]), np.array([]), {"reason": err or "missing_input_probe"}
        vinn, err = _complex_for_expr(df, input_neg_expr)
        if vinn is None:
            return np.array([]), np.array([]), {"reason": err or "missing_input_probe"}
        denom = vinp - vinn
    elif input_single_expr:
        vin, err = _complex_for_expr(df, input_single_expr)
        if vin is None:
            return np.array([]), np.array([]), {"reason": err or "missing_input_probe"}
        denom = vin
    else:
        denom = None

    if denom is None:
        resp = vout
    else:
        if np.any(np.abs(denom) < 1e-12):
            return np.array([]), np.array([]), {"reason": "input_zero"}
        resp = vout / denom

    order = np.argsort(freq)
    freq = freq[order]
    resp = resp[order]
    return freq, resp, None


def _mag_phase(resp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mag = np.abs(resp)
    mag = np.maximum(mag, _MAG_EPS)
    mag_db = 20 * np.log10(mag)
    phase_rad = np.unwrap(np.angle(resp))
    phase_deg = np.degrees(phase_rad)
    return mag_db, phase_deg


def _find_unity_crossing(freq: np.ndarray, mag_db: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    if freq.size < 2 or mag_db.size < 2:
        return None, {"reason": "insufficient_samples"}

    x = np.log10(freq)
    crossings: list[int] = []
    for i in range(len(mag_db) - 1):
        if mag_db[i] >= 0.0 and mag_db[i + 1] < 0.0:
            crossings.append(i)
            break

    if not crossings:
        if mag_db[0] < 0.0:
            return None, {"reason": "no_unity_crossing"}
        return None, {"reason": "unity_crossing_out_of_range"}

    i = crossings[0]
    y1 = mag_db[i]
    y2 = mag_db[i + 1]
    x1 = x[i]
    x2 = x[i + 1]
    if y2 == y1:
        x_cross = x1
    else:
        x_cross = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
    ugbw = float(10 ** x_cross)

    warnings: list[str] = []
    for j in range(i + 1, len(mag_db) - 1):
        if mag_db[j] >= 0.0 and mag_db[j + 1] < 0.0:
            warnings.append("multiple_crossings")
            break

    debug = {
        "bracket": {"f1": float(freq[i]), "f2": float(freq[i + 1]), "mag1_db": float(y1), "mag2_db": float(y2)},
        "warnings": warnings,
    }
    return ugbw, debug


def _interp_on_log_x(freq: np.ndarray, values: np.ndarray, target_freq: float) -> float:
    x = np.log10(freq)
    target_x = np.log10(target_freq)
    return float(np.interp(target_x, x, values))


def compute_ugbw_hz(raw: RawSimData, spec: MetricImplSpec) -> tuple[float | None, dict]:
    freq, resp, err = _prepare_transfer(raw, spec)
    if err:
        return _missing(err["reason"], debug={k: v for k, v in err.items() if k != "reason"})
    mag_db, _ = _mag_phase(resp)
    ugbw, debug = _find_unity_crossing(freq, mag_db)
    if ugbw is None:
        return _missing(debug.get("reason", "no_unity_crossing"), debug=debug)
    details: dict[str, Any] = {"status": "ok"}
    details.update({"debug": {"ugbw_hz": ugbw, **debug}})
    return ugbw, details


def compute_phase_margin_deg(raw: RawSimData, spec: MetricImplSpec) -> tuple[float | None, dict]:
    freq, resp, err = _prepare_transfer(raw, spec)
    if err:
        return _missing(err["reason"], debug={k: v for k, v in err.items() if k != "reason"})
    mag_db, phase_deg = _mag_phase(resp)
    ugbw, debug = _find_unity_crossing(freq, mag_db)
    if ugbw is None:
        return _missing(debug.get("reason", "no_unity_crossing"), debug=debug)
    phase_at_ugbw = _interp_on_log_x(freq, phase_deg, ugbw)
    pm = 180.0 + phase_at_ugbw
    details: dict[str, Any] = {
        "status": "ok",
        "debug": {
            "ugbw_hz": ugbw,
            "phase_deg_at_ugbw": float(phase_at_ugbw),
            **debug,
        },
    }
    return float(pm), details
