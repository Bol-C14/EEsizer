from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, runtime_checkable

from .artifacts import CircuitSource, CircuitSpec, ParamSpace, MetricsBundle, Patch


@dataclass
class Observation:
    """What a policy can see at one iteration."""
    spec: CircuitSpec
    source: CircuitSource
    param_space: ParamSpace
    metrics: MetricsBundle
    iteration: int
    history_tail: list[Dict[str, Any]] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Policy(Protocol):
    name: str
    version: str

    def propose(self, obs: Observation, ctx: Any) -> Patch:
        ...
