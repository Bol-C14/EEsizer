from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
import json
import os
import tempfile

from ..contracts.hashes import cfg_payload, hash_cfg, hash_spec, spec_payload
from ..contracts.session import PhaseRecord, SessionState
from ..contracts.strategy import StrategyConfig
from ..contracts.trace import TraceEntry
from .recorder import RunRecorder


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = path.parent
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(tmp_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp_path).replace(path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


class SessionStore:
    """Deterministic session storage under <run_dir>/session/."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.recorder = RunRecorder(self.run_dir)
        self.session_dir = self.run_dir / "session"
        self.checkpoints_dir = self.session_dir / "checkpoints"
        self.spec_revs_dir = self.session_dir / "spec_revs"
        self.cfg_revs_dir = self.session_dir / "cfg_revs"

    @property
    def state_path(self) -> Path:
        return self.session_dir / "session_state.json"

    @property
    def spec_trace_path(self) -> Path:
        return self.session_dir / "spec_trace.jsonl"

    @property
    def cfg_trace_path(self) -> Path:
        return self.session_dir / "cfg_trace.jsonl"

    def create_session(
        self,
        *,
        session_id: str,
        bench_id: str,
        seed: int | None,
        spec: Any,
        cfg: StrategyConfig,
        actor: str = "system",
        reason: str = "init",
    ) -> SessionState:
        created_at = _utc_now_iso()
        spec_h = hash_spec(spec)
        cfg_h = hash_cfg(cfg)
        state = SessionState(
            session_id=session_id,
            bench_id=bench_id,
            seed=seed,
            created_at=created_at,
            current_phase=None,
            spec_rev=0,
            cfg_rev=0,
            spec_hash=spec_h,
            cfg_hash=cfg_h,
            phases=[],
            artifacts_index={},
        )
        self.write_session_state(state)

        # Persist revision snapshots for reproducibility.
        self.write_spec_revision(rev=0, spec=spec)
        self.write_cfg_revision(rev=0, cfg=cfg)

        self.append_trace(
            TraceEntry(
                kind="spec",
                rev=0,
                timestamp=created_at,
                actor=actor,
                reason=reason,
                old_hash=None,
                new_hash=spec_h,
                diff={},
                linked_phase=None,
            )
        )
        self.append_trace(
            TraceEntry(
                kind="cfg",
                rev=0,
                timestamp=created_at,
                actor=actor,
                reason=reason,
                old_hash=None,
                new_hash=cfg_h,
                diff={},
                linked_phase=None,
            )
        )
        return state

    def load_session_state(self) -> SessionState:
        if not self.state_path.exists():
            raise FileNotFoundError(str(self.state_path))
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("invalid session_state.json")
        return SessionState.from_dict(payload)

    def write_session_state(self, state: SessionState) -> None:
        payload = json.dumps(state.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        _atomic_write_text(self.state_path, payload)

    def update_session_state(self, mut_fn: Callable[[SessionState], SessionState]) -> SessionState:
        state = self.load_session_state()
        new_state = mut_fn(state)
        self.write_session_state(new_state)
        return new_state

    def write_spec_revision(self, *, rev: int, spec: Any) -> Path:
        rel = f"session/spec_revs/spec_rev{rev:04d}.json"
        return self.recorder.write_json(rel, spec_payload(spec))

    def write_cfg_revision(self, *, rev: int, cfg: StrategyConfig) -> Path:
        rel = f"session/cfg_revs/cfg_rev{rev:04d}.json"
        return self.recorder.write_json(rel, cfg_payload(cfg))

    def append_trace(self, entry: TraceEntry) -> Path:
        if entry.kind == "cfg":
            return self.recorder.append_jsonl("session/cfg_trace.jsonl", entry.to_dict())
        return self.recorder.append_jsonl("session/spec_trace.jsonl", entry.to_dict())

    def write_checkpoint(self, phase_id: str, payload: Mapping[str, Any]) -> Path:
        safe_id = str(phase_id).strip()
        if not safe_id:
            raise ValueError("phase_id required")
        return self.recorder.write_json(f"session/checkpoints/{safe_id}.json", dict(payload))

    def load_checkpoint(self, phase_id: str) -> dict[str, Any] | None:
        path = self.session_dir / "checkpoints" / f"{phase_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def record_phase(
        self,
        *,
        phase_id: str,
        status: str,
        run_dir: str | None,
        spec_hash: str | None,
        cfg_hash: str | None,
        input_hash: str | None,
        output_summary: Mapping[str, Any] | None = None,
        stop_reason: str | None = None,
        notes: Mapping[str, Any] | None = None,
    ) -> None:
        record = PhaseRecord(
            phase_id=phase_id,
            status=status,
            run_dir=run_dir,
            spec_hash=spec_hash,
            cfg_hash=cfg_hash,
            input_hash=input_hash,
            output_summary=dict(output_summary or {}),
            stop_reason=stop_reason,
            notes=dict(notes or {}),
        )

        def _update(state: SessionState) -> SessionState:
            phases = [p for p in state.phases if p.phase_id != phase_id]
            phases.append(record)
            phases_sorted = sorted(phases, key=lambda p: p.phase_id)
            return replace(state, phases=phases_sorted, current_phase=phase_id)

        self.update_session_state(_update)

    # ---- LLM advice helpers (Step7) ----

    def advice_dir(self, rev: int) -> Path:
        return self.session_dir / "llm" / "advice" / f"advice_rev{int(rev):04d}"

    def latest_advice_rev(self) -> int | None:
        return self.load_session_state().latest_advice_rev

    def mark_advice_decision(self, *, rev: int, decision: str, reason: str | None = None) -> None:
        """Record accept/reject decision for an advice revision."""
        advice_dir = self.advice_dir(rev)
        status_path = advice_dir / "status.json"
        payload: dict[str, Any] = {}
        if status_path.exists():
            try:
                raw = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload.update(raw)
            except Exception:
                pass
        payload.update({"decision": str(decision), "decision_reason": reason, "decided_at": _utc_now_iso()})
        self.recorder.write_json(self.recorder.relpath(status_path), payload)

        def _update(state: SessionState) -> SessionState:
            history = list(state.advice_history)
            updated: list[dict[str, Any]] = []
            found = False
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                if int(entry.get("rev") or -1) == int(rev):
                    entry = dict(entry)
                    entry["decision"] = str(decision)
                    if reason is not None:
                        entry["decision_reason"] = reason
                    entry["decided_at"] = payload.get("decided_at")
                    found = True
                updated.append(entry)
            if not found:
                updated.append({"rev": int(rev), "decision": str(decision), "decision_reason": reason, "decided_at": payload.get("decided_at")})
            return replace(state, advice_history=updated)

        self.update_session_state(_update)

    # ---- Plan helpers (Step8) ----

    def plan_dir(self, rev: int) -> Path:
        return self.session_dir / "llm" / "plan_advice" / f"plan_rev{int(rev):04d}"

    def latest_plan_rev(self) -> int | None:
        return self.load_session_state().latest_plan_rev

    def mark_plan_decision(self, *, rev: int, decision: str, reason: str | None = None) -> None:
        plan_dir = self.plan_dir(rev)
        status_path = plan_dir / "status.json"
        payload: dict[str, Any] = {}
        if status_path.exists():
            try:
                raw = json.loads(status_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    payload.update(raw)
            except Exception:
                pass
        payload.update({"decision": str(decision), "decision_reason": reason, "decided_at": _utc_now_iso()})
        self.recorder.write_json(self.recorder.relpath(status_path), payload)

        def _update(state: SessionState) -> SessionState:
            history = list(state.plan_history)
            updated: list[dict[str, Any]] = []
            found = False
            for entry in history:
                if not isinstance(entry, dict):
                    continue
                if int(entry.get("rev") or -1) == int(rev):
                    entry = dict(entry)
                    entry["decision"] = str(decision)
                    if reason is not None:
                        entry["decision_reason"] = reason
                    entry["decided_at"] = payload.get("decided_at")
                    found = True
                updated.append(entry)
            if not found:
                updated.append(
                    {
                        "rev": int(rev),
                        "decision": str(decision),
                        "decision_reason": reason,
                        "decided_at": payload.get("decided_at"),
                    }
                )
            return replace(state, plan_history=updated)

        self.update_session_state(_update)
