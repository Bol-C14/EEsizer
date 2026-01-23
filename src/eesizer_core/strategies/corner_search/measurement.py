from __future__ import annotations

import re
from typing import Any, Mapping

from ...analysis.pareto import objective_losses
from ...contracts import MetricsBundle, Patch, PatchOp
from ...contracts.enums import PatchOpType
from ...contracts.guards import GuardReport
from ...runtime.recorder import RunRecorder
from ...runtime.recording_utils import guard_report_to_dict


_STAGE_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def stage_tag_for_corner(corner_id: str) -> str:
    text = str(corner_id or "").strip().lower()
    if not text:
        return "corner"
    return _STAGE_SAFE.sub("_", text)


def _metrics_values(metrics: MetricsBundle | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return {name: mv.value for name, mv in metrics.values.items()}


def _relativize_stage_map(stage_map: Mapping[str, str], recorder: RunRecorder | None) -> dict[str, str]:
    if recorder is None:
        return {str(k): str(v) for k, v in stage_map.items()}
    return {str(k): recorder.relpath(v) for k, v in stage_map.items()}


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

        base_val = base_values.get(param_id)
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
            bounds = param_bounds.get(param_id, {})
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            if lower is not None and new_val < float(lower):
                warnings.append(f"corner override '{param_id}' clamped to lower {lower}")
                new_val = float(lower)
            if upper is not None and new_val > float(upper):
                warnings.append(f"corner override '{param_id}' clamped to upper {upper}")
                new_val = float(upper)

        resolved[param_id] = float(new_val)

    return resolved, errors, warnings


def build_corner_patch(overrides: Mapping[str, float]) -> Patch:
    ops = [
        PatchOp(param=param_id, op=PatchOpType.set, value=value, why="corner_override")
        for param_id, value in overrides.items()
    ]
    return Patch(ops=tuple(ops))


def corner_result_dict(
    *,
    corner_id: str,
    overrides: Mapping[str, Any],
    metrics: MetricsBundle | None,
    eval_result: dict[str, Any] | None,
    stage_map: Mapping[str, str],
    warnings: list[str],
    errors: list[str],
    guard_report: GuardReport | None,
    recorder: RunRecorder | None,
) -> dict[str, Any]:
    metrics_payload = _metrics_values(metrics)
    losses = objective_losses(eval_result or {})
    score = eval_result.get("score", float("inf")) if eval_result is not None else float("inf")
    all_pass = bool(eval_result.get("all_pass")) if eval_result is not None else False
    return {
        "corner_id": corner_id,
        "overrides": dict(overrides),
        "metrics": metrics_payload,
        "score": score,
        "all_pass": all_pass,
        "losses": list(losses),
        "objectives": eval_result.get("per_objective", []) if eval_result is not None else [],
        "sim_stages": _relativize_stage_map(stage_map, recorder),
        "warnings": list(warnings),
        "errors": list(errors),
        "guard": guard_report_to_dict(guard_report) if guard_report else None,
    }


def pick_corner(
    corners: list[dict[str, Any]],
    *,
    corner_id: str | None = None,
    worst: bool = False,
) -> dict[str, Any] | None:
    if not corners:
        return None
    if corner_id is not None:
        for entry in corners:
            if entry.get("corner_id") == corner_id:
                return entry

    def _score(entry: Mapping[str, Any]) -> float:
        try:
            return float(entry.get("score", float("inf")))
        except (TypeError, ValueError):
            return float("inf")

    return max(corners, key=_score) if worst else min(corners, key=_score)
