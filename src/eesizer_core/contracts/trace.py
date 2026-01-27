from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TraceEntry:
    kind: str  # "spec" | "cfg"
    rev: int
    timestamp: str
    actor: str
    reason: str
    old_hash: str | None
    new_hash: str
    diff: dict[str, Any] = field(default_factory=dict)
    linked_phase: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rev": self.rev,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "reason": self.reason,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "diff": dict(self.diff),
            "linked_phase": self.linked_phase,
            "notes": dict(self.notes),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "TraceEntry":
        return TraceEntry(
            kind=str(payload.get("kind") or ""),
            rev=int(payload.get("rev") or 0),
            timestamp=str(payload.get("timestamp") or ""),
            actor=str(payload.get("actor") or ""),
            reason=str(payload.get("reason") or ""),
            old_hash=payload.get("old_hash"),
            new_hash=str(payload.get("new_hash") or ""),
            diff=dict(payload.get("diff") or {}),
            linked_phase=payload.get("linked_phase"),
            notes=dict(payload.get("notes") or {}),
        )

