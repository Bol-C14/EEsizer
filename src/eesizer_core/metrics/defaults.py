from __future__ import annotations

from ..contracts.enums import SimKind
from .registry import MetricRegistry, MetricImplSpec
from .ac import compute_ac_mag_db_at, compute_unity_gain_freq
from .dc import compute_dc_vout_last, compute_dc_slope
from .tran import compute_tran_rise_time

DEFAULT_REGISTRY = MetricRegistry(
    {
        "ac_mag_db_at_1k": MetricImplSpec(
            name="ac_mag_db_at_1k",
            unit="dB",
            requires_kind=SimKind.ac,
            requires_outputs=("ac_csv",),
            compute_fn=compute_ac_mag_db_at,
            params={"target_hz": 1e3, "node": "out"},
            description="Magnitude at 1kHz (dB) for node 'out'.",
        ),
        "ac_unity_gain_freq": MetricImplSpec(
            name="ac_unity_gain_freq",
            unit="Hz",
            requires_kind=SimKind.ac,
            requires_outputs=("ac_csv",),
            compute_fn=compute_unity_gain_freq,
            params={"node": "out", "target_db": 0.0},
            description="First frequency where gain crosses 0 dB for node 'out'.",
        ),
        "dc_vout_last": MetricImplSpec(
            name="dc_vout_last",
            unit="V",
            requires_kind=SimKind.dc,
            requires_outputs=("dc_csv",),
            compute_fn=compute_dc_vout_last,
            params={"node": "out"},
            description="DC output voltage at final sweep point for node 'out'.",
        ),
        "dc_slope": MetricImplSpec(
            name="dc_slope",
            unit="V/V",
            requires_kind=SimKind.dc,
            requires_outputs=("dc_csv",),
            compute_fn=compute_dc_slope,
            params={"sweep_col": "v(in)", "node": "out"},
            description="Approximate slope dV(out)/dV(in) over sweep range.",
        ),
        "tran_rise_time": MetricImplSpec(
            name="tran_rise_time",
            unit="s",
            requires_kind=SimKind.tran,
            requires_outputs=("tran_csv",),
            compute_fn=compute_tran_rise_time,
            params={"node": "out"},
            description="10-90%% rise time for node 'out'; may return None with diagnostics.",
        ),
    }
)
