from __future__ import annotations

from itertools import product
from math import isclose
from typing import Iterable, Mapping


def _coerce_float(value: float | int | None, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _unique_sorted(values: Iterable[float]) -> list[float]:
    ordered = []
    for val in sorted(values):
        if ordered and isclose(val, ordered[-1], rel_tol=1e-12, abs_tol=1e-12):
            continue
        ordered.append(val)
    return ordered


def make_levels(
    nominal: float,
    lower: float | None,
    upper: float | None,
    levels: int,
    span_mul: float,
    scale: str = "log",
) -> list[float]:
    if levels <= 0:
        return []
    nominal_f = float(nominal)
    span = float(span_mul) if span_mul > 0 else 1.0

    if lower is None or upper is None:
        if nominal_f == 0.0:
            lower_f = -span
            upper_f = span
        else:
            lower_f = nominal_f / span
            upper_f = nominal_f * span
    else:
        lower_f = float(lower)
        upper_f = float(upper)

    if lower_f > upper_f:
        lower_f, upper_f = upper_f, lower_f

    if levels == 1:
        if lower_f <= nominal_f <= upper_f:
            return [nominal_f]
        return [lower_f]

    scale_norm = str(scale or "log").lower()
    values: list[float]

    if scale_norm == "log":
        if lower_f <= 0 or upper_f <= 0:
            scale_norm = "linear"
        else:
            ratio = upper_f / lower_f
            if ratio <= 0:
                scale_norm = "linear"
            else:
                step = ratio ** (1.0 / (levels - 1))
                values = [lower_f * (step**idx) for idx in range(levels)]
                return _unique_sorted(values)

    step = (upper_f - lower_f) / (levels - 1)
    values = [lower_f + step * idx for idx in range(levels)]
    return _unique_sorted(values)


def coordinate_candidates(
    param_ids: Iterable[str],
    per_param_levels: Mapping[str, list[float]],
    baseline_values: Mapping[str, float],
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for param_id in param_ids:
        levels = per_param_levels.get(param_id)
        if not levels:
            continue
        baseline = baseline_values.get(param_id)
        for value in levels:
            if baseline is not None and isclose(value, baseline, rel_tol=1e-12, abs_tol=0.0):
                continue
            candidates.append({param_id: float(value)})
    return candidates


def factorial_candidates(
    param_ids: Iterable[str],
    per_param_levels: Mapping[str, list[float]],
    baseline_values: Mapping[str, float] | None = None,
    *,
    skip_baseline: bool = True,
) -> list[dict[str, float]]:
    ids = list(param_ids)
    if not ids:
        return []
    level_lists: list[list[float]] = []
    for pid in ids:
        levels = per_param_levels.get(pid)
        if not levels:
            return []
        level_lists.append(list(levels))

    candidates: list[dict[str, float]] = []
    for combo in product(*level_lists):
        candidate = {pid: float(val) for pid, val in zip(ids, combo)}
        if skip_baseline and baseline_values:
            if all(
                isclose(candidate.get(pid, 0.0), baseline_values.get(pid, 0.0), rel_tol=1e-12, abs_tol=0.0)
                for pid in ids
            ):
                continue
        candidates.append(candidate)
    return candidates
