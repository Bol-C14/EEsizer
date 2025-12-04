"""Reusable scoring and target helpers for sizing agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Callable, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class OptimizationTargets:
    """Design targets shared across agents."""

    gain_db: float
    power_mw: float


@dataclass(slots=True)
class ScoringPolicy:
    """Computes composite scores and stop criteria from normalized metrics."""

    targets: OptimizationTargets
    tolerance: float = 0.05
    weights: MutableMapping[str, float] = field(default_factory=lambda: {"gain": 1.0, "power": 1.0})
    plugins: Sequence[Callable[[Mapping[str, float]], float]] = field(default_factory=tuple)

    def score(self, metrics: Mapping[str, float]) -> float:
        """Geometric-mean score where gain is maximized and power is minimized."""

        factors: list[float] = []
        gain = metrics.get("gain_db")
        if gain is not None and self.targets.gain_db > 0:
            weight = float(self.weights.get("gain", 1.0))
            factors.append((max(float(gain), 0.0) / self.targets.gain_db) ** weight)
        power = metrics.get("power_mw")
        if power is not None and power > 0 and self.targets.power_mw > 0:
            weight = float(self.weights.get("power", 1.0))
            factors.append((self.targets.power_mw / max(float(power), 1e-9)) ** weight)
        for plugin in self.plugins:
            try:
                plugin_score = float(plugin(metrics))
            except Exception:
                plugin_score = 0.0
            if plugin_score > 0:
                factors.append(plugin_score)
        if not factors:
            return 0.0
        return float(prod(factors) ** (1 / len(factors)))

    def meets_targets(self, metrics: Mapping[str, float]) -> bool:
        """Check if metrics satisfy targets within tolerance."""

        gain = metrics.get("gain_db", 0.0)
        power = metrics.get("power_mw", 0.0)
        gain_ok = gain >= self.targets.gain_db * (1 - self.tolerance)
        power_ok = power <= self.targets.power_mw * (1 + self.tolerance)
        return gain_ok and power_ok

    def should_accept(self, current_score: float, candidate_score: float) -> bool:
        """Decide whether a candidate score is meaningfully better or equal."""

        return candidate_score >= current_score * (1 - self.tolerance)


__all__ = ["OptimizationTargets", "ScoringPolicy"]
