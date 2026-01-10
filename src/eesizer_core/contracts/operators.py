from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, runtime_checkable

from .provenance import Provenance


@dataclass
class OperatorResult:
    outputs: Dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=lambda: Provenance(operator="unknown"))
    logs: Dict[str, str] = field(default_factory=dict)  # small text logs
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Operator(Protocol):
    """Pure-ish transformation: inputs -> outputs with provenance."""
    name: str
    version: str

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        ...
