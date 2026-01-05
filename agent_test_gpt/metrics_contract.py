"""Metric contracts: definitions, units, and simulation requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class MetricError(RuntimeError):
    """Metric computation failed due to missing/invalid data."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    sim: str  # e.g., "ac", "dc", "tran"
    required_files: tuple[str, ...]
    description: str
    params: dict


# Canonical metric definitions used across code, prompts, and reporting.
METRICS: Mapping[str, MetricSpec] = {
    "gain_db": MetricSpec(
        name="gain_db",
        unit="dB",
        sim="ac",
        required_files=("output_ac.dat",),
        description="Low-frequency open-loop gain (median of first k AC samples in dB).",
        params={"median_first_k": 5},
    ),
    "bw_3db_hz": MetricSpec(
        name="bw_3db_hz",
        unit="Hz",
        sim="ac",
        required_files=("output_ac.dat",),
        description="-3 dB bandwidth: first frequency where gain drops 3 dB from low-frequency gain.",
        params={"drop_db": 3},
    ),
    "ugbw_hz": MetricSpec(
        name="ugbw_hz",
        unit="Hz",
        sim="ac",
        required_files=("output_ac.dat",),
        description="Unity-gain bandwidth: first 0 dB crossing of the AC gain.",
        params={},
    ),
    "out_swing_v": MetricSpec(
        name="out_swing_v",
        unit="V",
        sim="dc",
        required_files=("output_dc_ow.dat",),
        description="Linear output swing for a closed-loop gain target; computed from DC sweep with feedback network.",
        params={"gain_target_env": "OUT_SWING_GAIN_TARGET", "tol_env": "OUT_SWING_GAIN_TOL"},
    ),
}

