from __future__ import annotations

DEFAULT_TOL: dict[str, dict[str, float | None]] = {
    "gain_db": {"abs": 0.2, "rel": None},
    "phase_margin_deg": {"abs": 2.0, "rel": None},
    "pm_deg": {"abs": 2.0, "rel": None},
    "bw_3db_hz": {"abs": None, "rel": 0.05},
    "ugbw_hz": {"abs": None, "rel": 0.05},
    "out_swing_v": {"abs": 0.01, "rel": 0.05},
    "offset_v": {"abs": 1e-3, "rel": None},
    "icmr_v": {"abs": 1e-3, "rel": None},
    "tran_gain_db": {"abs": 0.2, "rel": None},
    "power_w": {"abs": None, "rel": 0.1},
}
