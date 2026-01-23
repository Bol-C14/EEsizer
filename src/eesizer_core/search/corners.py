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


def _override_for_bounds(
    *,
    nominal: float | None,
    lower: float,
    upper: float,
    override_mode: str,
    errors: list[str],
    param_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = str(override_mode or "add").lower()
    if mode not in {"set", "add", "mul"}:
        errors.append(f"param '{param_id}' unsupported override_mode '{override_mode}'")
        mode = "add"
    if mode == "set":
        return {"op": "set", "value": lower}, {"op": "set", "value": upper}
    if nominal is None:
        errors.append(f"param '{param_id}' missing nominal for {mode} override, falling back to set")
        return {"op": "set", "value": lower}, {"op": "set", "value": upper}
    if mode == "mul":
        if nominal == 0.0:
            errors.append(f"param '{param_id}' nominal=0 for mul override, falling back to set")
            return {"op": "set", "value": lower}, {"op": "set", "value": upper}
        return (
            {"op": "mul", "value": lower / nominal},
            {"op": "mul", "value": upper / nominal},
        )
    return (
        {"op": "add", "value": lower - nominal},
        {"op": "add", "value": upper - nominal},
    )


def build_corner_set(
    *,
    param_space: ParamSpace,
    nominal_values: Mapping[str, float],
    span_mul: float = 10.0,
    corner_param_ids: Iterable[str] | None = None,
    include_global_corners: bool = False,
    override_mode: str = "add",
    mode: str = "oat",
) -> dict[str, Any]:
    mode_norm = str(mode or "oat").lower()
    if mode_norm != "oat":
        raise ValueError(f"unsupported corner mode '{mode}'")

    ids = _normalize_param_ids(param_space, corner_param_ids)
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
    ]
    if include_global_corners:
        all_low: dict[str, Any] = {}
        all_high: dict[str, Any] = {}
        for param_id in active_ids:
            nominal = bounds[param_id]["nominal"]
            lower = bounds[param_id]["lower"]
            upper = bounds[param_id]["upper"]
            if lower is None or upper is None:
                continue
            low_override, high_override = _override_for_bounds(
                nominal=nominal,
                lower=float(lower),
                upper=float(upper),
                override_mode=override_mode,
                errors=errors,
                param_id=param_id,
            )
            all_low[param_id] = low_override
            all_high[param_id] = high_override
        corners.append({"corner_id": "all_low", "label": "all_low", "overrides": all_low})
        corners.append({"corner_id": "all_high", "label": "all_high", "overrides": all_high})
    for param_id in active_ids:
        nominal = bounds[param_id]["nominal"]
        lower = bounds[param_id]["lower"]
        upper = bounds[param_id]["upper"]
        if lower is None or upper is None:
            continue
        low_override, high_override = _override_for_bounds(
            nominal=nominal,
            lower=float(lower),
            upper=float(upper),
            override_mode=override_mode,
            errors=errors,
            param_id=param_id,
        )
        corners.append(
            {
                "corner_id": f"{param_id}_low",
                "label": f"{param_id}_low",
                "overrides": {param_id: low_override},
            }
        )
        corners.append(
            {
                "corner_id": f"{param_id}_high",
                "label": f"{param_id}_high",
                "overrides": {param_id: high_override},
            }
        )

    return {
        "mode": mode_norm,
        "span_mul": span_mul,
        "override_mode": str(override_mode or "add").lower(),
        "include_global_corners": include_global_corners,
        "param_ids": active_ids,
        "corner_param_ids": active_ids,
        "param_bounds": bounds,
        "corners": corners,
        "errors": errors,
    }
