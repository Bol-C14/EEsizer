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
    ) -> list[ExecutionEvent]:
        """Execute actions sequentially.

        Returns the list of execution events.
        """
        actions = validate_plan(plan)
        events: list[ExecutionEvent] = []

        for idx, action in enumerate(actions):
            if not self.registry.has(action.op):
                raise ValidationError(f"plan references unknown op '{action.op}'")

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
