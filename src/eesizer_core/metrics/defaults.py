from __future__ import annotations

from ..contracts.enums import SimKind
from .registry import MetricRegistry, MetricSpec
from .ac import compute_ac_mag_db_at, compute_unity_gain_freq

DEFAULT_REGISTRY = MetricRegistry(
    {
        "ac_mag_db_at_1k": MetricSpec(
            name="ac_mag_db_at_1k",
            unit="dB",
            requires_kind=SimKind.ac,
            requires_outputs=("ac_csv",),
            compute_fn=compute_ac_mag_db_at,
            params={"target_hz": 1e3, "node": "out"},
            description="Magnitude at 1kHz (dB) for node 'out'.",
        ),
        "ac_unity_gain_freq": MetricSpec(
            name="ac_unity_gain_freq",
            unit="Hz",
            requires_kind=SimKind.ac,
            requires_outputs=("ac_csv",),
            compute_fn=compute_unity_gain_freq,
            params={"node": "out", "target_db": 0.0},
            description="First frequency where gain crosses 0 dB for node 'out'.",
        ),
    }
)
