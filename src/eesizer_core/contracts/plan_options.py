from __future__ import annotations

from typing import Any, Mapping

from .errors import ValidationError


PLAN_OPTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["options"],
    "properties": {
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "plan"],
                "properties": {
                    "title": {"type": "string"},
                    "intent": {"type": "string"},
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "op", "inputs", "outputs", "params"],
                            "properties": {
                                "id": {"type": "string"},
                                "op": {"type": "string"},
                                "inputs": {"type": "array", "items": {"type": "string"}},
                                "outputs": {"type": "array", "items": {"type": "string"}},
                                "params": {"type": "object"},
                                "requires_approval": {"type": "boolean"},
                                "notes": {"type": "string"},
                                "stop_if": {"type": "object"},
                            },
                        },
                    },
                    "expected_effects": {"type": "array", "items": {"type": "string"}},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "budget_estimate": {
                        "type": "object",
                        "properties": {"iters": {"type": "integer"}, "corners": {"type": "integer"}},
                    },
                },
            },
        }
    },
}


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def validate_llm_plan_options(payload: Any) -> dict[str, Any]:
    """Static validation / normalization for PlanOptions.

    Semantic validation (tool allowlist, budgets, params bounds, etc.) is handled separately.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError("plan_options must be a JSON object")

    options_raw = payload.get("options")
    if not isinstance(options_raw, list) or not options_raw:
        raise ValidationError("plan_options.options must be a non-empty list")

    normalized_options: list[dict[str, Any]] = []
    for opt in options_raw:
        if not isinstance(opt, Mapping):
            continue
        title = opt.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValidationError("plan_options.options[].title must be a non-empty string")

        plan_raw = opt.get("plan")
        if not isinstance(plan_raw, list) or not plan_raw:
            raise ValidationError("plan_options.options[].plan must be a non-empty list")

        seen_ids: set[str] = set()
        seen_outputs: set[str] = set()
        plan: list[dict[str, Any]] = []
        for item in plan_raw:
            if not isinstance(item, Mapping):
                continue
            aid = item.get("id")
            op = item.get("op")
            if not isinstance(aid, str) or not aid.strip():
                raise ValidationError("plan action.id must be a non-empty string")
            if aid in seen_ids:
                raise ValidationError(f"duplicate action id: {aid}")
            seen_ids.add(aid)
            if not isinstance(op, str) or not op.strip():
                raise ValidationError("plan action.op must be a non-empty string")

            inputs = _as_str_list(item.get("inputs"))
            outputs = _as_str_list(item.get("outputs"))
            for out_name in outputs:
                if out_name in seen_outputs:
                    raise ValidationError(f"duplicate plan output name: {out_name}")
                seen_outputs.add(out_name)

            params = item.get("params") or {}
            if not isinstance(params, Mapping):
                raise ValidationError("plan action.params must be an object")

            plan.append(
                {
                    "id": aid.strip(),
                    "op": op.strip(),
                    "inputs": inputs,
                    "outputs": outputs,
                    "params": dict(params),
                    "requires_approval": bool(item.get("requires_approval", False)),
                    "notes": str(item.get("notes") or ""),
                    "stop_if": dict(item.get("stop_if") or {}) if isinstance(item.get("stop_if"), Mapping) else None,
                }
            )

        normalized_options.append(
            {
                "title": title.strip(),
                "intent": str(opt.get("intent") or "").strip(),
                "plan": plan,
                "expected_effects": _as_str_list(opt.get("expected_effects")),
                "risks": _as_str_list(opt.get("risks")),
                "budget_estimate": dict(opt.get("budget_estimate") or {}) if isinstance(opt.get("budget_estimate"), Mapping) else None,
            }
        )

    if not normalized_options:
        raise ValidationError("plan_options.options must contain at least one option object")

    return {"options": normalized_options}

