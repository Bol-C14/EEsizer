from __future__ import annotations

from typing import Any, Mapping


_ALIASES: dict[str, tuple[str, str | None]] = {
    # Gain
    "gain_output": ("gain_db", "dB"),
    "gain": ("gain_db", "dB"),
    "gain_db": ("gain_db", "dB"),
    # Phase margin
    "pm": ("phase_margin_deg", "deg"),
    "pm_output": ("phase_margin_deg", "deg"),
    "phase_margin": ("phase_margin_deg", "deg"),
    "pm_deg": ("phase_margin_deg", "deg"),
    "phase_margin_deg": ("phase_margin_deg", "deg"),
    # Bandwidth
    "bw": ("bw_3db_hz", "Hz"),
    "bw_output": ("bw_3db_hz", "Hz"),
    "bw_3db_hz": ("bw_3db_hz", "Hz"),
    "ugbw_hz": ("ugbw_hz", "Hz"),
    "ubw_output": ("ugbw_hz", "Hz"),
    "unity_bandwidth": ("ugbw_hz", "Hz"),
    "ac_unity_gain_freq": ("ugbw_hz", "Hz"),
    # Output swing / offsets
    "out_swing_v": ("out_swing_v", "V"),
    "ow_output": ("out_swing_v", "V"),
    "output_swing": ("out_swing_v", "V"),
    "offset_output": ("offset_v", "V"),
    "offset_v": ("offset_v", "V"),
    "icmr_output": ("icmr_v", "V"),
    "icmr": ("icmr_v", "V"),
    "icmr_v": ("icmr_v", "V"),
    # Transient / power
    "tr_gain_output": ("tran_gain_db", "dB"),
    "tran_gain_db": ("tran_gain_db", "dB"),
    "power": ("power_w", "W"),
    "pr_output": ("power_w", "W"),
    "power_w": ("power_w", "W"),
    # CMRR / THD
    "cmrr_output": ("cmrr_db", "dB"),
    "cmrr": ("cmrr_db", "dB"),
    "cmrr_db": ("cmrr_db", "dB"),
    "thd_output": ("thd_ratio", None),
    "thd": ("thd_ratio", None),
    "thd_ratio": ("thd_ratio", None),
}


def canonicalize_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonicalized metrics dict with unified names and units."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        canonical, unit = _ALIASES.get(key, (key, None))
        if isinstance(value, Mapping):
            payload = dict(value)
            if unit is not None:
                payload["unit"] = unit
            out[canonical] = payload
        else:
            out[canonical] = value
    return out


def canonicalize_metric_name(name: str) -> str:
    canonical, _ = _ALIASES.get(name, (name, None))
    return canonical
