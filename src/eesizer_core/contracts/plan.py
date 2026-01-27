"""Structured execution plans for multi-stage strategies.

The intent is to keep LLM/agents constrained to proposing *data* (Plans/Patches),
while deterministic Operators/Strategies do all side-effectful work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Sequence, Tuple

from .errors import ValidationError


@dataclass(frozen=True)
class Action:
    """One step in a Plan.

    - op: operator/tool name registered in a ToolRegistry
    - inputs: artifact names to fetch from an ArtifactStore
    - outputs: artifact names to store back into the ArtifactStore
    - params: JSON-friendly parameters for the op
    """

    op: str
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    params: Dict[str, Any] = field(default_factory=dict)
    # Optional metadata (Step8): kept separate from params to stay audit-friendly.
    id: str | None = None
    requires_approval: bool = False
    notes: str | None = None


Plan = Tuple[Action, ...]


def _ensure_str_seq(values: Iterable[Any], *, what: str) -> Tuple[str, ...]:
    out: list[str] = []
    for v in values:
        if not isinstance(v, str) or not v.strip():
            raise ValidationError(f"{what} must be non-empty strings")
        out.append(v)
    return tuple(out)


def validate_action(action: Action) -> None:
    if not isinstance(action, Action):
        raise ValidationError("action must be an Action")
    if not isinstance(action.op, str) or not action.op.strip():
        raise ValidationError("action.op must be a non-empty string")
    _ensure_str_seq(action.inputs, what="action.inputs")
    outs = _ensure_str_seq(action.outputs, what="action.outputs")
    if len(set(outs)) != len(outs):
        raise ValidationError("action.outputs must be unique")
    if not isinstance(action.params, dict):
        raise ValidationError("action.params must be a dict")
    if action.id is not None and (not isinstance(action.id, str) or not action.id.strip()):
        raise ValidationError("action.id must be a non-empty string when provided")
    if not isinstance(action.requires_approval, bool):
        raise ValidationError("action.requires_approval must be a bool")
    if action.notes is not None and not isinstance(action.notes, str):
        raise ValidationError("action.notes must be a string when provided")


def validate_plan(plan: Sequence[Action]) -> Plan:
    if not isinstance(plan, (list, tuple)):
        raise ValidationError("plan must be a list/tuple of Action")
    seen_outputs: set[str] = set()
    actions: list[Action] = []
    for idx, action in enumerate(plan):
        try:
            validate_action(action)
        except ValidationError as exc:
            raise ValidationError(f"invalid action[{idx}]: {exc}") from exc
        for out_name in action.outputs:
            if out_name in seen_outputs:
                raise ValidationError(f"duplicate plan output name: {out_name}")
            seen_outputs.add(out_name)
        actions.append(action)
    return tuple(actions)
