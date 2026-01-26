from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
import random

from ..contracts.grid_search_config import GridSearchConfig
from ..contracts.artifacts import ParamSpace
from ..search.samplers import coordinate_candidates, factorial_candidates, make_levels


@dataclass(frozen=True)
class RangeTrace:
    param_id: str
    nominal: float | None
    lower: float | None
    upper: float | None
    scale: str
    levels: list[float]
    source: str
    span_mul: float | None
    sanity: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_id": self.param_id,
            "nominal": self.nominal,
            "lower": self.lower,
            "upper": self.upper,
            "scale": self.scale,
            "levels": list(self.levels),
            "source": self.source,
            "span_mul": self.span_mul,
            "sanity": dict(self.sanity),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


def infer_ranges(
    param_ids: Iterable[str],
    param_space: ParamSpace,
    nominal_values: Mapping[str, float],
    cfg: GridSearchConfig,
) -> list[RangeTrace]:
    traces: list[RangeTrace] = []
    for raw_pid in param_ids:
        pid = raw_pid.lower()
        param_def = param_space.get(pid)
        if param_def is None:
            traces.append(
                RangeTrace(
                    param_id=pid,
                    nominal=None,
                    lower=None,
                    upper=None,
                    scale=cfg.scale_for(pid),
                    levels=[],
                    source="unknown",
                    span_mul=None,
                    sanity={"warnings": []},
                    skipped=True,
                    skip_reason="param_not_found",
                )
            )
            continue

        nominal = nominal_values.get(pid)
        span_mul = cfg.span_mul_for(pid)
        lower = param_def.lower
        upper = param_def.upper
        source = ""
        warnings: list[str] = []
        clipped = False

        if lower is not None and upper is not None:
            source = "bounds"
        elif nominal is not None:
            span = span_mul if span_mul > 0 else 1.0
            derived_lower = nominal / span if nominal != 0 else -span
            derived_upper = nominal * span if nominal != 0 else span
            if lower is None:
                lower = derived_lower
            if upper is None:
                upper = derived_upper
            source = "nominal*span_mul" if param_def.lower is None and param_def.upper is None else "bounds+span"
        else:
            traces.append(
                RangeTrace(
                    param_id=pid,
                    nominal=None,
                    lower=None,
                    upper=None,
                    scale=cfg.scale_for(pid),
                    levels=[],
                    source="unknown",
                    span_mul=span_mul,
                    sanity={"warnings": []},
                    skipped=True,
                    skip_reason="missing_nominal",
                )
            )
            continue

        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
            clipped = True
            warnings.append("bounds_swapped")

        if nominal is not None and lower is not None and upper is not None:
            if nominal < lower or nominal > upper:
                warnings.append("nominal_outside_bounds")

        scale = cfg.scale_for(pid)
        if scale == "log" and (lower is None or upper is None or lower <= 0 or upper <= 0):
            scale = "linear"
            warnings.append("log_fallback_linear")

        levels_count = cfg.levels_for(pid)
        levels = []
        if lower is not None and upper is not None:
            levels = make_levels(
                nominal=nominal if nominal is not None else 0.0,
                lower=lower,
                upper=upper,
                levels=levels_count,
                span_mul=span_mul,
                scale=scale,
                include_nominal=cfg.include_nominal and nominal is not None,
            )

        traces.append(
            RangeTrace(
                param_id=pid,
                nominal=nominal,
                lower=lower,
                upper=upper,
                scale=scale,
                levels=levels,
                source=source,
                span_mul=span_mul,
                sanity={
                    "positive_required": scale == "log",
                    "clipped": clipped,
                    "warnings": warnings,
                },
            )
        )
    return traces


def generate_candidates(
    ranges: Sequence[RangeTrace],
    baseline_values: Mapping[str, float],
    cfg: GridSearchConfig,
    *,
    max_candidates: int | None,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    param_ids = [r.param_id for r in ranges if not r.skipped and r.levels]
    per_param_levels = {r.param_id: list(r.levels) for r in ranges if not r.skipped and r.levels}

    if cfg.mode == "factorial":
        candidates = factorial_candidates(
            param_ids,
            per_param_levels,
            baseline_values,
            include_nominal=cfg.include_nominal,
        )
    else:
        candidates = coordinate_candidates(
            param_ids,
            per_param_levels,
            baseline_values,
            include_nominal=cfg.include_nominal,
        )

    total_generated = len(candidates)
    truncated = False
    if max_candidates is not None and total_generated > max_candidates:
        truncated = True
        if cfg.truncate_policy == "seed_shuffle":
            rng = random.Random(cfg.seed)
            rng.shuffle(candidates)
        candidates = candidates[:max_candidates]

    meta = {
        "mode": cfg.mode,
        "include_nominal": cfg.include_nominal,
        "total_generated": total_generated,
        "truncated_to": len(candidates),
        "max_candidates": max_candidates,
        "truncate_policy": cfg.truncate_policy,
        "seed": cfg.seed,
        "truncated": truncated,
    }
    return candidates, meta
