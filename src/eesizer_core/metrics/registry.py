from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from ..contracts.enums import SimKind
from ..contracts.errors import ValidationError
from ..sim.artifacts import RawSimData

ComputeFn = Callable[[RawSimData, "MetricSpec"], float | None]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    requires_kind: SimKind
    requires_outputs: tuple[str, ...]
    compute_fn: ComputeFn
    params: dict[str, object] = field(default_factory=dict)
    description: str = ""


class MetricRegistry:
    """Lookup and resolve metric specs by name."""

    def __init__(self, specs: Mapping[str, MetricSpec]) -> None:
        self._specs: dict[str, MetricSpec] = dict(specs)

    def get(self, name: str) -> MetricSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise ValidationError(f"Unknown metric '{name}'")
        return spec

    def list(self) -> list[str]:
        return sorted(self._specs.keys())

    def resolve(self, names: Iterable[str]) -> list[MetricSpec]:
        resolved = []
        for name in names:
            resolved.append(self.get(name))
        return resolved
