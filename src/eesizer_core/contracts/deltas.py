from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ObjectiveDelta:
    metric: str
    op: str  # target|weight|tol|sense|add|remove
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "op": self.op, "value": self.value}

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "ObjectiveDelta":
        return ObjectiveDelta(
            metric=str(payload.get("metric") or "").strip(),
            op=str(payload.get("op") or "").strip().lower(),
            value=payload.get("value"),
        )


@dataclass(frozen=True)
class SpecDelta:
    objectives: tuple[ObjectiveDelta, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objectives": [o.to_dict() for o in self.objectives],
            "notes": dict(self.notes),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "SpecDelta":
        obj_raw = payload.get("objectives") or []
        objectives: list[ObjectiveDelta] = []
        if isinstance(obj_raw, list):
            for item in obj_raw:
                if isinstance(item, Mapping):
                    objectives.append(ObjectiveDelta.from_dict(item))
        notes = payload.get("notes") or {}
        notes_dict = dict(notes) if isinstance(notes, Mapping) else {}
        return SpecDelta(objectives=tuple(objectives), notes=notes_dict)


@dataclass(frozen=True)
class CfgDelta:
    budget: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"budget": dict(self.budget), "seed": self.seed, "notes": dict(self.notes)}

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "CfgDelta":
        budget = payload.get("budget") or {}
        notes = payload.get("notes") or {}
        return CfgDelta(
            budget=dict(budget) if isinstance(budget, Mapping) else {},
            seed=payload.get("seed"),
            notes=dict(notes) if isinstance(notes, Mapping) else {},
        )

