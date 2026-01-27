from __future__ import annotations

from typing import Any, Mapping

from .deltas import CfgDelta, SpecDelta
from .errors import ValidationError


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["options"],
    "properties": {
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "intent", "expected_effects", "risks", "budget_estimate"],
                "properties": {
                    "title": {"type": "string"},
                    "intent": {"type": "string"},
                    "spec_delta": {"type": ["object", "null"]},
                    "cfg_delta": {"type": ["object", "null"]},
                    "plan": {"type": ["array", "null"]},
                    "expected_effects": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "budget_estimate": {
                        "type": "object",
                        "required": ["iters", "corners"],
                        "properties": {
                            "iters": {"type": "integer"},
                            "corners": {"type": "integer"},
                        },
                    },
                },
            },
        }
    },
}


_ALLOWED_SPEC_OPS = {"target", "weight", "tol", "sense", "add", "remove"}
_ALLOWED_CFG_NOTES_TOPLEVEL = {"grid_search", "corner_validate"}
_ALLOWED_BUDGET_KEYS = {"max_iterations", "max_sim_runs", "timeout_s", "no_improve_patience"}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _validate_spec_delta(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValidationError("proposal.spec_delta must be an object or null")
    delta = SpecDelta.from_dict(payload)
    for obj in delta.objectives:
        if obj.op not in _ALLOWED_SPEC_OPS:
            raise ValidationError(f"proposal.spec_delta objective op '{obj.op}' is not allowed")
    if delta.notes:
        raise ValidationError("proposal.spec_delta.notes is not allowed (objective-only deltas)")
    return delta.to_dict()


def _validate_cfg_delta(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValidationError("proposal.cfg_delta must be an object or null")
    delta = CfgDelta.from_dict(payload)
    # Budget: only allow known fields (apply_cfg_delta ignores unknowns, but we want the
    # proposal contract to be an explicit safety boundary).
    unknown_budget = sorted(set(delta.budget.keys()) - _ALLOWED_BUDGET_KEYS, key=str)
    if unknown_budget:
        raise ValidationError(f"proposal.cfg_delta.budget contains unsupported keys: {unknown_budget}")

    # Notes: restrict to a small allowlist to avoid accidental tool / runtime overrides.
    unknown_notes = sorted(set(delta.notes.keys()) - _ALLOWED_CFG_NOTES_TOPLEVEL, key=str)
    if unknown_notes:
        raise ValidationError(f"proposal.cfg_delta.notes contains unsupported keys: {unknown_notes}")
    return delta.to_dict()


def validate_llm_proposal(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValidationError("proposal must be a JSON object")
    options = payload.get("options")
    if not isinstance(options, list) or not options:
        raise ValidationError("proposal.options must be a non-empty list")

    normalized: list[dict[str, Any]] = []
    for opt in options:
        if not isinstance(opt, Mapping):
            continue
        title = opt.get("title")
        intent = opt.get("intent")
        if not isinstance(title, str) or not title.strip():
            raise ValidationError("proposal.options[].title must be a string")
        if not isinstance(intent, str) or not intent.strip():
            raise ValidationError("proposal.options[].intent must be a string")

        budget = opt.get("budget_estimate") or {}
        if not isinstance(budget, Mapping):
            raise ValidationError("proposal.options[].budget_estimate must be an object")
        iters = budget.get("iters")
        corners = budget.get("corners")
        if not isinstance(iters, int) or not isinstance(corners, int):
            raise ValidationError("proposal.options[].budget_estimate.{iters,corners} must be integers")

        spec_delta = _validate_spec_delta(opt.get("spec_delta"))
        cfg_delta = _validate_cfg_delta(opt.get("cfg_delta"))

        plan = opt.get("plan")
        if plan is not None and not isinstance(plan, list):
            raise ValidationError("proposal.options[].plan must be a list or null")

        normalized.append(
            {
                "title": title.strip(),
                "intent": intent.strip(),
                "spec_delta": spec_delta,
                "cfg_delta": cfg_delta,
                "plan": plan,
                "expected_effects": _as_str_list(opt.get("expected_effects")),
                "risks": _as_str_list(opt.get("risks")),
                "budget_estimate": {"iters": int(iters), "corners": int(corners)},
            }
        )

    if not normalized:
        raise ValidationError("proposal.options must contain at least one object option")

    return {"options": normalized}
