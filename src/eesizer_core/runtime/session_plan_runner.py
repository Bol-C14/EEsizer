from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from ..contracts.errors import ValidationError
from ..contracts.plan import Action
from ..contracts.provenance import stable_hash_json
from .artifact_store import ArtifactStore
from .plan_executor import PlanExecutor
from .recorder import RunRecorder
from .tool_registry import ToolRegistry


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def plan_dir_for_rev(session_run_dir: Path, plan_rev: int) -> Path:
    return Path(session_run_dir) / "session" / "llm" / "plan_advice" / f"plan_rev{int(plan_rev):04d}"


def actions_from_plan_option(plan_option: Mapping[str, Any]) -> list[Action]:
    raw = plan_option.get("plan")
    if not isinstance(raw, list) or not raw:
        raise ValidationError("plan option missing 'plan'")
    actions: list[Action] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        actions.append(
            Action(
                op=str(item.get("op") or ""),
                inputs=tuple(item.get("inputs") or ()),
                outputs=tuple(item.get("outputs") or ()),
                params=dict(item.get("params") or {}),
                id=str(item.get("id") or "") or None,
                requires_approval=bool(item.get("requires_approval", False)),
                notes=str(item.get("notes") or "") or None,
            )
        )
    if not actions:
        raise ValidationError("plan option has no valid actions")
    return actions


@dataclass(frozen=True)
class PlanExecutionState:
    status: str  # pending|running|paused|completed|error|aborted
    next_action_idx: int
    last_event: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "next_action_idx": int(self.next_action_idx),
            "last_event": dict(self.last_event or {}),
        }


def load_execution_state(plan_dir: Path) -> PlanExecutionState | None:
    payload = _read_json(plan_dir / "execution_state.json")
    if not isinstance(payload, Mapping):
        return None
    try:
        return PlanExecutionState(
            status=str(payload.get("status") or "pending"),
            next_action_idx=int(payload.get("next_action_idx") or 0),
            last_event=dict(payload.get("last_event") or {}) if isinstance(payload.get("last_event"), Mapping) else None,
        )
    except Exception:
        return None


def write_execution_state(plan_dir: Path, state: PlanExecutionState) -> None:
    _write_json(plan_dir / "execution_state.json", state.to_dict())


def write_dry_run(
    *,
    plan_dir: Path,
    plan_option: Mapping[str, Any],
    tool_catalog: Mapping[str, Any] | None = None,
) -> Path:
    title = str(plan_option.get("title") or "plan").strip()
    lines: list[str] = []
    lines.append("# Plan Dry Run")
    lines.append("")
    lines.append(f"- title: {title}")
    lines.append(f"- actions: {len(plan_option.get('plan') or [])}")
    if tool_catalog and isinstance(tool_catalog.get("sha256"), str):
        lines.append(f"- tool_catalog_sha256: {tool_catalog.get('sha256')}")
    lines.append("")
    lines.append("## Actions")
    tools_by_name: dict[str, Mapping[str, Any]] = {}
    if tool_catalog and isinstance(tool_catalog.get("tools"), list):
        for t in tool_catalog["tools"]:
            if isinstance(t, Mapping) and isinstance(t.get("name"), str):
                tools_by_name[t["name"]] = t

    for act in plan_option.get("plan") or []:
        if not isinstance(act, Mapping):
            continue
        aid = act.get("id")
        op = act.get("op")
        req = bool(act.get("requires_approval", False))
        lines.append(f"- {aid} :: {op} (requires_approval={req})")
        tmeta = tools_by_name.get(str(op))
        if tmeta:
            desc = str(tmeta.get("description") or "").strip()
            if desc:
                lines.append(f"  - {desc}")
            cm = tmeta.get("cost_model") or {}
            if cm:
                lines.append(f"  - cost_model: {json.dumps(cm, sort_keys=True)}")
            se = tmeta.get("side_effects") or []
            if se:
                lines.append(f"  - side_effects: {json.dumps(se, sort_keys=True)}")

    out = plan_dir / "dry_run.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def execute_plan_option(
    *,
    plan_dir: Path,
    actions: list[Action],
    registry: ToolRegistry,
    session_run_dir: Path,
    source: Any,
    auto_approve: bool = False,
    resume: bool = True,
) -> PlanExecutionState:
    recorder = RunRecorder(Path(session_run_dir))
    root_dir = recorder.relpath(plan_dir / "artifacts")
    store = ArtifactStore(recorder, root_dir=root_dir)
    store.put("session_run_dir", Path(session_run_dir))
    store.put("source", source)

    # Resume from saved state.
    start_idx = 0
    prev = load_execution_state(plan_dir)
    if resume and prev is not None:
        start_idx = int(prev.next_action_idx)

    write_execution_state(plan_dir, PlanExecutionState(status="running", next_action_idx=start_idx))

    exe = PlanExecutor(registry)
    try:
        events = exe.execute(
            actions,
            store=store,
            ctx=None,
            recorder=recorder,
            log_path=recorder.relpath(plan_dir / "execution.jsonl"),
            start_idx=start_idx,
            stop_before_approval=not auto_approve,
        )
    except Exception as exc:
        state = PlanExecutionState(status="error", next_action_idx=start_idx, last_event={"error": str(exc)})
        write_execution_state(plan_dir, state)
        raise

    last = events[-1] if events else None
    if last is not None and last.status == "paused":
        state = PlanExecutionState(status="paused", next_action_idx=int(last.action_idx), last_event=last.to_dict())
        write_execution_state(plan_dir, state)
        return state

    # If we started in the middle, we need a best-effort next_action_idx.
    max_idx = start_idx + len([e for e in events if e.status == "ok"])
    state = PlanExecutionState(status="completed", next_action_idx=max(len(actions), max_idx), last_event=last.to_dict() if last else None)
    write_execution_state(plan_dir, state)
    return state


def stable_plan_hash(plan_option: Mapping[str, Any]) -> str:
    return stable_hash_json(dict(plan_option))

