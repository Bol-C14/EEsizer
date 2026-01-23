from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol

from ..contracts import CircuitIR, CircuitSource, CircuitSpec, StrategyConfig


@dataclass(frozen=True)
class AgentContext:
    """Inputs an Agent is allowed to read.

    Agents must not do side-effectful work (no file IO, no ngspice runs). They
    only propose JSON-friendly artifacts and configs.
    """

    source: CircuitSource
    circuit_ir: CircuitIR
    signature: str
    cfg: StrategyConfig
    spec: CircuitSpec | None = None
    extras: Mapping[str, Any] | None = None


class Agent(Protocol):
    name: str
    version: str

    def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a mapping of artifact_name -> artifact_value."""
        ...
