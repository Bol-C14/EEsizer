from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CornerSearchConfig:
    mode: str
    levels: int
    span_mul: float
    scale: str
    top_k: int
    stop_on_first_pass: bool
    baseline_retries: int
    corner_mode: str
    include_global_corners: bool
    override_mode: str
    require_baseline_corner_pass: bool
    clamp_corner_overrides: bool
    search_param_ids: list[str] | None
    corner_param_ids: list[str] | None
