from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class GridSearchConfig:
    mode: str
    levels: int
    span_mul: float
    scale: str
    top_k: int
    stop_on_first_pass: bool
    baseline_retries: int
    max_params: int
    max_candidates: int | None
    param_ids: tuple[str, ...] = ()
    recommended_knobs: tuple[str, ...] = ()
    param_select_policy: str = "recommended"
    truncate_policy: str = "seed_shuffle"
    include_nominal: bool = False
    allow_param_ids_override_frozen: bool = False
    seed: int = 0
    per_param_levels: Mapping[str, int] = field(default_factory=dict)
    per_param_span_mul: Mapping[str, float] = field(default_factory=dict)
    per_param_scale: Mapping[str, str] = field(default_factory=dict)

    def levels_for(self, param_id: str) -> int:
        return self.per_param_levels.get(param_id.lower(), self.levels)

    def span_mul_for(self, param_id: str) -> float:
        return self.per_param_span_mul.get(param_id.lower(), self.span_mul)

    def scale_for(self, param_id: str) -> str:
        return self.per_param_scale.get(param_id.lower(), self.scale)
