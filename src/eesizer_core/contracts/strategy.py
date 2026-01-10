from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from .artifacts import CircuitSource, CircuitSpec, RunResult
from .enums import StopReason


@dataclass(frozen=True)
class OptimizationBudget:
    max_iterations: int = 25
    max_sim_runs: Optional[int] = None
    timeout_s: Optional[float] = None
    no_improve_patience: int = 5


@dataclass
class StrategyConfig:
    budget: OptimizationBudget = field(default_factory=OptimizationBudget)
    seed: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    name: str
    version: str

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:
        ...
