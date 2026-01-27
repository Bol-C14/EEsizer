from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts.deltas import CfgDelta, SpecDelta
from ..contracts.errors import ValidationError
from ..runtime.tool_registry import ToolRegistry


DEFAULT_PLAN_GUARDRAILS: dict[str, Any] = {
    "max_actions_per_option": 12,
    "max_budget_max_iterations": 200,
    "grid_search": {"levels_min": 3, "levels_max": 15, "span_mul_min": 1.2, "span_mul_max": 50.0, "top_k_max": 20},
}


def _validate_range_int(name: str, value: Any, *, lo: int, hi: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, int):
        return [f"{name} must be an integer"]
    if value < lo or value > hi:
        return [f"{name} must be in [{lo}, {hi}]"]
    return []


def _validate_range_num(name: str, value: Any, *, lo: float, hi: float) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (int, float)):
        return [f"{name} must be a number"]
    v = float(value)
    if v < lo or v > hi:
        return [f"{name} must be in [{lo}, {hi}]"]
    return []


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _validate_spec_delta(payload: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if payload is None:
        return None, []
    raw = _as_mapping(payload)
    if raw is None:
        return None, ["spec_delta must be an object"]
    delta = SpecDelta.from_dict(raw)
    allowed_ops = {"target", "weight", "tol", "sense", "add", "remove"}
    for obj in delta.objectives:
        if obj.op not in allowed_ops:
            return None, [f"spec_delta objective op '{obj.op}' is not allowed"]
    if delta.notes:
        return None, ["spec_delta.notes is not allowed"]
    return delta.to_dict(), []


def _validate_cfg_delta(payload: Any, guardrails: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    if payload is None:
        return None, []
    raw = _as_mapping(payload)
    if raw is None:
        return None, ["cfg_delta must be an object"]
    delta = CfgDelta.from_dict(raw)

    errors: list[str] = []

    allowed_budget = {"max_iterations", "max_sim_runs", "timeout_s", "no_improve_patience"}
    bad_budget = sorted(set(delta.budget.keys()) - allowed_budget, key=str)
    if bad_budget:
        errors.append(f"cfg_delta.budget contains unsupported keys: {bad_budget}")

    allowed_notes_top = {"grid_search", "corner_validate"}
    bad_notes = sorted(set(delta.notes.keys()) - allowed_notes_top, key=str)
    if bad_notes:
        errors.append(f"cfg_delta.notes contains unsupported keys: {bad_notes}")

    max_max_iters = int(guardrails.get("max_budget_max_iterations", 200))
    if "max_iterations" in delta.budget:
        try:
            mi = int(delta.budget.get("max_iterations"))
        except Exception:
            errors.append("cfg_delta.budget.max_iterations must be an integer")
        else:
            if mi < 1 or mi > max_max_iters:
                errors.append(f"cfg_delta.budget.max_iterations must be in [1, {max_max_iters}]")

    # Common grid_search guardrails.
    grid_limits = guardrails.get("grid_search") if isinstance(guardrails.get("grid_search"), Mapping) else {}
    grid = delta.notes.get("grid_search") if isinstance(delta.notes.get("grid_search"), Mapping) else {}
    if isinstance(grid, Mapping):
        errors += _validate_range_int(
            "cfg_delta.notes.grid_search.levels",
            grid.get("levels"),
            lo=int(grid_limits.get("levels_min", 3)),
            hi=int(grid_limits.get("levels_max", 15)),
        )
        errors += _validate_range_num(
            "cfg_delta.notes.grid_search.span_mul",
            grid.get("span_mul"),
            lo=float(grid_limits.get("span_mul_min", 1.2)),
            hi=float(grid_limits.get("span_mul_max", 50.0)),
        )
        errors += _validate_range_int(
            "cfg_delta.notes.grid_search.top_k",
            grid.get("top_k"),
            lo=1,
            hi=int(grid_limits.get("top_k_max", 20)),
        )

    if errors:
        return None, errors

    return delta.to_dict(), []


@dataclass(frozen=True)
class PlanValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


def validate_plan_options_semantics(
    plan_options: Mapping[str, Any],
    *,
    registry: ToolRegistry,
    guardrails: Mapping[str, Any] | None = None,
) -> PlanValidationReport:
    """Semantic validation for already-parsed plan_options (dict).

    Checks:
    - tool op exists
    - action inputs/outputs follow ToolSpec.io when provided
    - cfg_delta/spec_delta payloads are constrained
    - basic size limits (max actions)
    """
    guard = dict(DEFAULT_PLAN_GUARDRAILS)
    if guardrails:
        guard.update(dict(guardrails))

    errors: list[str] = []
    warnings: list[str] = []

    options = plan_options.get("options") if isinstance(plan_options.get("options"), list) else []
    if not options:
        return PlanValidationReport(ok=False, errors=["options missing/empty"], warnings=[])

    max_actions = int(guard.get("max_actions_per_option", 12))

    for opt_idx, opt in enumerate(options):
        if not isinstance(opt, Mapping):
            continue
        plan = opt.get("plan") if isinstance(opt.get("plan"), list) else []
        if len(plan) > max_actions:
            errors.append(f"option[{opt_idx}] has too many actions ({len(plan)} > {max_actions})")
        for act_idx, act in enumerate(plan):
            if not isinstance(act, Mapping):
                continue
            op = str(act.get("op") or "")
            if not registry.has(op):
                errors.append(f"option[{opt_idx}].plan[{act_idx}] unknown op '{op}'")
                continue

            spec = registry.spec(op)
            expected_io = spec.io or {}
            expected_inputs = list(expected_io.get("inputs") or [])
            expected_outputs = list(expected_io.get("outputs") or [])
            inputs = act.get("inputs") if isinstance(act.get("inputs"), list) else []
            outputs = act.get("outputs") if isinstance(act.get("outputs"), list) else []
            if expected_inputs and inputs != expected_inputs:
                errors.append(f"option[{opt_idx}].plan[{act_idx}] inputs must be {expected_inputs}")
            if expected_outputs and outputs != expected_outputs:
                errors.append(f"option[{opt_idx}].plan[{act_idx}] outputs must be {expected_outputs}")

            params = act.get("params") if isinstance(act.get("params"), Mapping) else {}
            if op == "update_spec":
                _, errs = _validate_spec_delta(params.get("spec_delta"))
                errors += [f"option[{opt_idx}].plan[{act_idx}] {e}" for e in errs]
            elif op == "update_cfg":
                _, errs = _validate_cfg_delta(params.get("cfg_delta"), guard)
                errors += [f"option[{opt_idx}].plan[{act_idx}] {e}" for e in errs]

    return PlanValidationReport(ok=not errors, errors=errors, warnings=warnings)


def raise_on_invalid_plan(report: PlanValidationReport) -> None:
    if not report.ok:
        raise ValidationError("invalid plan: " + "; ".join(report.errors[:5]))

