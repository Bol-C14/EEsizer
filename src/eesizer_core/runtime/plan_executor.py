"""Execute a structured Plan against an ArtifactStore using a ToolRegistry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence
import time

from ..contracts.plan import Action, validate_plan
from ..contracts.errors import ValidationError
from .artifact_store import ArtifactStore
from .recorder import RunRecorder
from .tool_registry import ToolRegistry


@dataclass
class ExecutionEvent:
    action_idx: int
    action_id: str | None
    op: str
    status: str  # "ok"|"error"
    start_time: float
    end_time: float
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    error: str | None = None
    params: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_idx": self.action_idx,
            "action_id": self.action_id,
            "op": self.op,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": max(0.0, self.end_time - self.start_time),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "params": dict(self.params or {}),
            "error": self.error,
        }


class PlanExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def execute(
        self,
        plan: Sequence[Action],
        *,
        store: ArtifactStore,
        ctx: Any,
        recorder: RunRecorder | None = None,
        log_path: str = "orchestrator/plan_execution.jsonl",
        start_idx: int = 0,
        stop_before_approval: bool = False,
    ) -> list[ExecutionEvent]:
        """Execute actions sequentially.

        Returns the list of execution events.
        """
        actions = validate_plan(plan)
        events: list[ExecutionEvent] = []

        for idx, action in enumerate(actions):
            if idx < int(start_idx):
                continue
            if not self.registry.has(action.op):
                raise ValidationError(f"plan references unknown op '{action.op}'")

            if stop_before_approval and action.requires_approval:
                now = time.time()
                ev = ExecutionEvent(
                    action_idx=idx,
                    action_id=action.id,
                    op=action.op,
                    status="paused",
                    start_time=now,
                    end_time=now,
                    inputs={},
                    outputs={},
                    params=dict(action.params),
                    error="requires_approval",
                )
                events.append(ev)
                if recorder is not None:
                    recorder.append_jsonl(log_path, ev.to_dict())
                break

            start = time.time()
            in_objs: Dict[str, Any] = {}
            in_hashes: Dict[str, str] = {}
            for name in action.inputs:
                in_objs[name] = store.get(name)
                try:
                    in_hashes[name] = store.entry(name).get("sha256", "")
                except Exception:
                    in_hashes[name] = ""

            out_hashes: Dict[str, str] = {}
            try:
                params = dict(action.params)
                params.setdefault("_action_idx", idx)
                params.setdefault("_action_id", action.id)
                params.setdefault("_action_outputs", list(action.outputs))
                result = self.registry.execute(action.op, in_objs, ctx, params)

                # Ensure all declared outputs are present.
                for out_name in action.outputs:
                    if out_name not in result:
                        raise ValidationError(f"tool '{action.op}' did not produce output '{out_name}'")
                    store.put(out_name, result[out_name], producer=action.op)
                    out_hashes[out_name] = store.entry(out_name).get("sha256", "")

                end = time.time()
                ev = ExecutionEvent(
                    action_idx=idx,
                    action_id=action.id,
                    op=action.op,
                    status="ok",
                    start_time=start,
                    end_time=end,
                    inputs=in_hashes,
                    outputs=out_hashes,
                    params=dict(action.params),
                )
                events.append(ev)
                if recorder is not None:
                    recorder.append_jsonl(log_path, ev.to_dict())
            except Exception as exc:
                end = time.time()
                ev = ExecutionEvent(
                    action_idx=idx,
                    action_id=action.id,
                    op=action.op,
                    status="error",
                    start_time=start,
                    end_time=end,
                    inputs=in_hashes,
                    outputs=out_hashes,
                    error=f"{type(exc).__name__}: {exc}",
                    params=dict(action.params),
                )
                events.append(ev)
                if recorder is not None:
                    recorder.append_jsonl(log_path, ev.to_dict())
                raise

        return events
