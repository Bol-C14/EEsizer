"""Simulation helpers used by agents and tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from typing import MutableMapping

from .config import SimulationConfig
from .netlist import summarize_components


@dataclass(slots=True)
class SimulationResult:
    metrics: MutableMapping[str, float]


class MockNgSpiceSimulator:
    """Deterministic, dependency-free simulator stub."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(self, netlist_text: str) -> MutableMapping[str, float]:
        counts = summarize_components(netlist_text)
        mos = counts.get("M", 0)
        resistors = counts.get("R", 0)
        capacitors = counts.get("C", 0)
        diodes = counts.get("D", 0)

        gain_db = 20.0 + mos * 4.5
        power_mw = max(0.1, 0.4 + resistors * 0.15 + mos * 0.05)
        bandwidth_hz = 1e6 / (1 + capacitors)
        noise_mv = 0.25 + max(0.0, 2.0 - log10(1 + mos + diodes))

        return {
            "gain_db": gain_db,
            "power_mw": round(power_mw, 3),
            "bandwidth_hz": round(bandwidth_hz, 3),
            "noise_mv": round(noise_mv, 3),
            "transistor_count": float(mos),
        }


__all__ = ["MockNgSpiceSimulator", "SimulationResult"]
