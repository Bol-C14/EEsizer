from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Tuple

GuardSeverity = Literal["hard", "soft"]


@dataclass(frozen=True)
class GuardCheck:
    name: str
    ok: bool
    severity: GuardSeverity
    reasons: Tuple[str, ...] = ()
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


@dataclass(frozen=True)
class GuardReport:
    checks: Tuple[GuardCheck, ...] = ()
    ok: bool = True
    hard_fails: Tuple[GuardCheck, ...] = ()
    soft_fails: Tuple[GuardCheck, ...] = ()

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        hard_fails = tuple(c for c in checks if c.severity == "hard" and not c.ok)
        soft_fails = tuple(c for c in checks if c.severity == "soft" and not c.ok)
        ok = not hard_fails
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "hard_fails", hard_fails)
        object.__setattr__(self, "soft_fails", soft_fails)
        object.__setattr__(self, "ok", ok)
