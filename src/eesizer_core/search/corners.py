from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts import ParamSpace


def _bound_from_nominal(nominal: float | None, span_mul: float, *, is_lower: bool) -> float | None:
    if nominal is None:
        return None
    span = float(span_mul) if span_mul > 0 else 1.0
    if nominal == 0.0:
        return -span if is_lower else span
    return nominal / span if is_lower else nominal * span


def _resolve_bounds(
    param_id: str,
    param_space: ParamSpace,
    nominal_values: Mapping[str, float],
    *,
    span_mul: float,
) -> tuple[float | None, float | None, float | None]:
    param_def = param_space.get(param_id)
    nominal = nominal_values.get(param_id)
    lower = param_def.lower if param_def is not None else None
    upper = param_def.upper if param_def is not None else None
    if lower is None:
        lower = _bound_from_nominal(nominal, span_mul, is_lower=True)
    if upper is None:
        upper = _bound_from_nominal(nominal, span_mul, is_lower=False)
    if lower is not None and upper is not None and lower > upper:
        lower, upper = upper, lower
    return nominal, lower, upper


def _normalize_param_ids(param_space: ParamSpace, param_ids: Iterable[str] | None) -> list[str]:
    if param_ids is None:
        return [p.param_id.lower() for p in param_space.params]
    return [str(pid).lower() for pid in param_ids]


def build_corner_set(
    *,
    param_space: ParamSpace,
    nominal_values: Mapping[str, float],
    span_mul: float = 10.0,
    param_ids: Iterable[str] | None = None,
    mode: str = "oat",
) -> dict[str, Any]:
    mode_norm = str(mode or "oat").lower()
    if mode_norm != "oat":
        raise ValueError(f"unsupported corner mode '{mode}'")

    ids = _normalize_param_ids(param_space, param_ids)
    bounds: dict[str, dict[str, float | None]] = {}
    errors: list[str] = []
    for param_id in ids:
        nominal, lower, upper = _resolve_bounds(
            param_id,
            param_space,
            nominal_values,
            span_mul=span_mul,
        )
        if lower is None or upper is None:
            errors.append(f"param '{param_id}' missing bounds/nominal for corners")
            continue
        bounds[param_id] = {"nominal": nominal, "lower": float(lower), "upper": float(upper)}

    active_ids = sorted(bounds.keys())
    corners: list[dict[str, Any]] = [
        {"corner_id": "nominal", "label": "nominal", "overrides": {}},
        {"corner_id": "all_low", "label": "all_low", "overrides": {pid: bounds[pid]["lower"] for pid in active_ids}},
        {"corner_id": "all_high", "label": "all_high", "overrides": {pid: bounds[pid]["upper"] for pid in active_ids}},
    ]
    for param_id in active_ids:
        corners.append(
            {
                "corner_id": f"{param_id}_low",
                "label": f"{param_id}_low",
                "overrides": {param_id: bounds[param_id]["lower"]},
            }
        )
        corners.append(
            {
                "corner_id": f"{param_id}_high",
                "label": f"{param_id}_high",
                "overrides": {param_id: bounds[param_id]["upper"]},
            }
        )

    return {
        "mode": mode_norm,
        "span_mul": span_mul,
        "param_ids": active_ids,
        "param_bounds": bounds,
        "corners": corners,
        "errors": errors,
    }
