from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

from ..contracts import CircuitSource
from ..contracts.errors import ValidationError
from ..strategies.interactive_session import InteractiveSessionStrategy
from .session_store import SessionStore


def _require_path(value: Any, name: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    raise ValidationError(f"missing required Path input: {name}")


def _require_source(value: Any) -> CircuitSource:
    if isinstance(value, CircuitSource):
        return value
    raise ValidationError("missing required CircuitSource input: source")


def build_session_plan_tool_fns(*, measure_fn: Any = None) -> dict[str, Any]:
    """Concrete tool implementations for Step8 session plans.

    All tools expect inputs:
      - session_run_dir: Path
      - source: CircuitSource
    """

    strat = InteractiveSessionStrategy(measure_fn=measure_fn)

    def update_spec(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_run_dir = _require_path(inputs.get("session_run_dir"), "session_run_dir")
        source = _require_source(inputs.get("source"))
        spec_delta = params.get("spec_delta")
        if not isinstance(spec_delta, Mapping):
            raise ValidationError("update_spec requires params.spec_delta object")
        strat.continue_session(
            session_run_dir=session_run_dir,
            source=source,
            spec_delta=dict(spec_delta),
            next_phase=None,
            actor="llm",
            reason=str(params.get("reason") or "plan:update_spec"),
        )
        state = SessionStore(session_run_dir).load_session_state()
        return {"session/spec_rev": state.spec_rev, "session/spec_hash": state.spec_hash}

    def update_cfg(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_run_dir = _require_path(inputs.get("session_run_dir"), "session_run_dir")
        source = _require_source(inputs.get("source"))
        cfg_delta = params.get("cfg_delta")
        if not isinstance(cfg_delta, Mapping):
            raise ValidationError("update_cfg requires params.cfg_delta object")
        strat.continue_session(
            session_run_dir=session_run_dir,
            source=source,
            cfg_delta=dict(cfg_delta),
            next_phase=None,
            actor="llm",
            reason=str(params.get("reason") or "plan:update_cfg"),
        )
        state = SessionStore(session_run_dir).load_session_state()
        return {"session/cfg_rev": state.cfg_rev, "session/cfg_hash": state.cfg_hash}

    def run_baseline(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_run_dir = _require_path(inputs.get("session_run_dir"), "session_run_dir")
        source = _require_source(inputs.get("source"))
        strat.continue_session(
            session_run_dir=session_run_dir,
            source=source,
            next_phase="p0_baseline",
            actor="system",
            reason=str(params.get("reason") or "plan:run_baseline"),
        )
        ck = SessionStore(session_run_dir).load_checkpoint("p0_baseline") or {}
        return {"p0/run_dir": ck.get("run_dir")}

    def run_grid_search(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_run_dir = _require_path(inputs.get("session_run_dir"), "session_run_dir")
        source = _require_source(inputs.get("source"))
        strat.continue_session(
            session_run_dir=session_run_dir,
            source=source,
            next_phase="p1_grid",
            actor="system",
            reason=str(params.get("reason") or "plan:run_grid_search"),
        )
        ck = SessionStore(session_run_dir).load_checkpoint("p1_grid") or {}
        run_dir = ck.get("run_dir")
        topk = []
        pareto = []
        if isinstance(run_dir, str) and run_dir:
            try:
                topk_path = Path(run_dir) / "search" / "topk.json"
                pareto_path = Path(run_dir) / "search" / "pareto.json"
                if topk_path.exists():
                    topk = json.loads(topk_path.read_text(encoding="utf-8"))
                if pareto_path.exists():
                    pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
            except Exception:
                topk = []
                pareto = []
        return {"p1/run_dir": run_dir, "p1/topk": topk, "p1/pareto": pareto}

    def run_corner_validate(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
        session_run_dir = _require_path(inputs.get("session_run_dir"), "session_run_dir")
        source = _require_source(inputs.get("source"))
        strat.continue_session(
            session_run_dir=session_run_dir,
            source=source,
            next_phase="p2_corner_validate",
            actor="system",
            reason=str(params.get("reason") or "plan:run_corner_validate"),
        )
        ck = SessionStore(session_run_dir).load_checkpoint("p1_grid") or {}
        run_dir = ck.get("run_dir")
        robust_topk = []
        robust_pareto = []
        if isinstance(run_dir, str) and run_dir:
            try:
                topk_path = Path(run_dir) / "search" / "robust_topk.json"
                pareto_path = Path(run_dir) / "search" / "robust_pareto.json"
                if topk_path.exists():
                    robust_topk = json.loads(topk_path.read_text(encoding="utf-8"))
                if pareto_path.exists():
                    robust_pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
            except Exception:
                robust_topk = []
                robust_pareto = []
        return {"p2/run_dir": run_dir, "p2/robust_topk": robust_topk, "p2/robust_pareto": robust_pareto}

    return {
        "update_spec": update_spec,
        "update_cfg": update_cfg,
        "run_baseline": run_baseline,
        "run_grid_search": run_grid_search,
        "run_corner_validate": run_corner_validate,
    }

