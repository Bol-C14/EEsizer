from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PhaseRecord:
    phase_id: str
    status: str  # completed|skipped|failed
    run_dir: str | None = None
    spec_hash: str | None = None
    cfg_hash: str | None = None
    input_hash: str | None = None
    output_summary: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "status": self.status,
            "run_dir": self.run_dir,
            "spec_hash": self.spec_hash,
            "cfg_hash": self.cfg_hash,
            "input_hash": self.input_hash,
            "output_summary": dict(self.output_summary),
            "stop_reason": self.stop_reason,
            "notes": dict(self.notes),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "PhaseRecord":
        return PhaseRecord(
            phase_id=str(payload.get("phase_id") or ""),
            status=str(payload.get("status") or ""),
            run_dir=payload.get("run_dir"),
            spec_hash=payload.get("spec_hash"),
            cfg_hash=payload.get("cfg_hash"),
            input_hash=payload.get("input_hash"),
            output_summary=dict(payload.get("output_summary") or {}),
            stop_reason=payload.get("stop_reason"),
            notes=dict(payload.get("notes") or {}),
        )


@dataclass(frozen=True)
class SessionState:
    session_id: str
    bench_id: str
    seed: int | None = None
    created_at: str | None = None
    current_phase: str | None = None
    spec_rev: int = 0
    cfg_rev: int = 0
    spec_hash: str | None = None
    cfg_hash: str | None = None
    phases: list[PhaseRecord] = field(default_factory=list)
    artifacts_index: dict[str, str] = field(default_factory=dict)
    latest_advice_rev: int | None = None
    advice_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "bench_id": self.bench_id,
            "seed": self.seed,
            "created_at": self.created_at,
            "current_phase": self.current_phase,
            "spec_rev": self.spec_rev,
            "cfg_rev": self.cfg_rev,
            "spec_hash": self.spec_hash,
            "cfg_hash": self.cfg_hash,
            "phases": [p.to_dict() for p in self.phases],
            "artifacts_index": dict(self.artifacts_index),
            "latest_advice_rev": self.latest_advice_rev,
            "advice_history": list(self.advice_history),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "SessionState":
        phases_raw = payload.get("phases") or []
        phases: list[PhaseRecord] = []
        if isinstance(phases_raw, list):
            for item in phases_raw:
                if isinstance(item, Mapping):
                    phases.append(PhaseRecord.from_dict(item))
        return SessionState(
            session_id=str(payload.get("session_id") or ""),
            bench_id=str(payload.get("bench_id") or ""),
            seed=payload.get("seed"),
            created_at=payload.get("created_at"),
            current_phase=payload.get("current_phase"),
            spec_rev=int(payload.get("spec_rev") or 0),
            cfg_rev=int(payload.get("cfg_rev") or 0),
            spec_hash=payload.get("spec_hash"),
            cfg_hash=payload.get("cfg_hash"),
            phases=phases,
            artifacts_index=dict(payload.get("artifacts_index") or {}),
            latest_advice_rev=payload.get("latest_advice_rev"),
            advice_history=list(payload.get("advice_history") or []),
        )
