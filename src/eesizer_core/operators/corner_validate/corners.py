from __future__ import annotations

from typing import Any, Mapping

from ...contracts import Patch, PatchOp
from ...contracts.enums import PatchOpType
from ...runtime.stage_names import sanitize_stage_name


def stage_tag_for_corner(corner_id: str, *, prefix: str = "cv") -> str:
    return sanitize_stage_name(f"{prefix}_{corner_id}", default=prefix)


def build_candidate_patch(candidate: Mapping[str, Any]) -> Patch:
    ops = []
    for param_id, value in candidate.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        ops.append(PatchOp(param=str(param_id).lower(), op=PatchOpType.set, value=numeric, why="candidate"))
    return Patch(ops=tuple(ops))


def resolve_corner_overrides(
    *,
    base_values: Mapping[str, float],
    param_bounds: Mapping[str, Mapping[str, float | None]],
    overrides: Mapping[str, Any],
    clamp: bool,
) -> tuple[dict[str, float], list[str], list[str]]:
    resolved: dict[str, float] = {}
    errors: list[str] = []
    warnings: list[str] = []

    for param_id, raw in overrides.items():
        op = "set"
        value = raw
        if isinstance(raw, Mapping):
            op = str(raw.get("op", "set")).lower()
            value = raw.get("value")

        if op not in {"set", "add", "mul"}:
            errors.append(f"corner override for '{param_id}' has unsupported op '{op}'")
            continue
        if value is None:
            errors.append(f"corner override for '{param_id}' missing value")
            continue

        base_val = base_values.get(str(param_id).lower())
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"corner override for '{param_id}' has non-numeric value '{value}'")
            continue

        if op == "set":
            new_val = numeric
        elif base_val is None:
            errors.append(f"corner override for '{param_id}' missing base value for op '{op}'")
            continue
        elif op == "add":
            new_val = base_val + numeric
        else:
            new_val = base_val * numeric

        if clamp:
            bounds = param_bounds.get(str(param_id).lower(), {})
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            if lower is not None and new_val < float(lower):
                warnings.append(f"corner override '{param_id}' clamped to lower {lower}")
                new_val = float(lower)
            if upper is not None and new_val > float(upper):
                warnings.append(f"corner override '{param_id}' clamped to upper {upper}")
                new_val = float(upper)

        resolved[str(param_id).lower()] = float(new_val)

    return resolved, errors, warnings


def build_corner_patch(overrides: Mapping[str, float]) -> Patch:
    ops = [
        PatchOp(param=param_id, op=PatchOpType.set, value=value, why="corner_override")
        for param_id, value in overrides.items()
    ]
    return Patch(ops=tuple(ops))

