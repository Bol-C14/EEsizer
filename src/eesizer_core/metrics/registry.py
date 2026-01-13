from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from ..contracts.enums import SimKind
from ..contracts.errors import ValidationError
from ..sim.artifacts import RawSimData
ComputeFn = Callable[[RawSimData, "MetricImplSpec"], tuple[float | None, dict]]


@dataclass(frozen=True)
class MetricImplSpec:
    """Executable metric definition (implementation-level)."""

    name: str
    unit: str
    requires_kind: SimKind
    requires_outputs: tuple[str, ...]
    compute_fn: ComputeFn
    params: dict[str, object] = field(default_factory=dict)
    description: str = ""


class MetricRegistry:
    """Lookup and resolve metric specs by name."""

    def __init__(self, specs: Mapping[str, MetricImplSpec]) -> None:
        normalized: dict[str, MetricImplSpec] = {}
        for name, spec in specs.items():
            if spec.name != name:
                raise ValidationError(f"Metric name mismatch: key '{name}' vs spec.name '{spec.name}'")
            if name in normalized:
                raise ValidationError(f"Duplicate metric '{name}' in registry")
            normalized[name] = spec
        self._specs = normalized

    def get(self, name: str) -> MetricImplSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise ValidationError(f"Unknown metric '{name}'")
        return spec

    def available(self) -> list[str]:
        return sorted(self._specs.keys())

    def resolve(self, names: Iterable[str]) -> list[MetricImplSpec]:
        resolved = []
        for name in names:
            resolved.append(self.get(name))
        return resolved
