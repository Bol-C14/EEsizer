from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json

from ..analysis.session_report import build_meta_report
from ..contracts import CircuitSource, CircuitSpec, Constraint, Objective
from ..contracts.deltas import CfgDelta, SpecDelta
from ..contracts.errors import ValidationError
from ..contracts.hashes import cfg_payload, hash_cfg, hash_payload, hash_spec, spec_payload
from ..contracts.session import SessionState
from ..contracts.strategy import OptimizationBudget, StrategyConfig
from ..contracts.trace import TraceEntry
from ..operators.apply_cfg_delta import apply_cfg_delta
from ..operators.apply_spec_delta import apply_spec_delta
from ..operators.spec_diff import diff_specs
from ..runtime.context import RunContext
from ..runtime.recorder import RunRecorder
from ..runtime.session_store import SessionStore
from .baseline_noopt import NoOptBaselineStrategy
from .grid_search import GridSearchStrategy
from ..operators.corner_validate import CornerValidateOperator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _deep_merge(base: Any, updates: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(updates, Mapping):
        out = dict(base)
        for k, v in updates.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return updates


def _cfg_delta_from_mapping(payload: Mapping[str, Any]) -> CfgDelta:
    raw = dict(payload)
    notes_delta = dict(raw.get("notes") or {}) if isinstance(raw.get("notes"), Mapping) else {}
    for k, v in raw.items():
        if k in {"budget", "seed", "notes"}:
            continue
        notes_delta[k] = v
    return CfgDelta.from_dict(
        {
            "budget": raw.get("budget") or {},
            "seed": raw.get("seed"),
            "notes": notes_delta,
        }
    )


def _diff_cfg(old: StrategyConfig, new: StrategyConfig) -> dict[str, Any]:
    old_p = cfg_payload(old)
    new_p = cfg_payload(new)

    budget_old = old_p.get("budget") or {}
    budget_new = new_p.get("budget") or {}
    budget_changed: list[dict[str, Any]] = []
    if isinstance(budget_old, Mapping) and isinstance(budget_new, Mapping):
        keys = sorted(set(budget_old.keys()) | set(budget_new.keys()))
        for k in keys:
            if budget_old.get(k) != budget_new.get(k):
                budget_changed.append({"field": k, "from": budget_old.get(k), "to": budget_new.get(k)})

    notes_old = old_p.get("notes") if isinstance(old_p.get("notes"), Mapping) else {}
    notes_new = new_p.get("notes") if isinstance(new_p.get("notes"), Mapping) else {}
    notes_changed_keys = sorted(
        {k for k in notes_old.keys() ^ notes_new.keys()}
        | {k for k in notes_old.keys() & notes_new.keys() if notes_old.get(k) != notes_new.get(k)},
        key=str,
    )

    return {"budget_changed": budget_changed, "notes_changed_keys": notes_changed_keys}


def _build_spec(payload: Mapping[str, Any]) -> CircuitSpec:
    objectives: list[Objective] = []
    for obj in payload.get("objectives", []) or []:
        if not isinstance(obj, Mapping):
            continue
        metric = obj.get("metric")
        if not metric:
            continue
        objectives.append(
            Objective(
                metric=str(metric),
                target=obj.get("target"),
                tol=obj.get("tol"),
                weight=float(obj.get("weight", 1.0)),
                sense=str(obj.get("sense", "ge")),
            )
        )

    constraints: list[Constraint] = []
    for c in payload.get("constraints", []) or []:
        if not isinstance(c, Mapping):
            continue
        kind = c.get("kind")
        data = c.get("data") or {}
        if not kind or not isinstance(data, Mapping):
            continue
        constraints.append(Constraint(kind=str(kind), data=dict(data)))

    observables = []
    for name in payload.get("observables", []) or []:
        if isinstance(name, str) and name.strip():
            observables.append(name.strip())

    notes = payload.get("notes") or {}
    notes_dict = dict(notes) if isinstance(notes, Mapping) else {}
    return CircuitSpec(
        objectives=tuple(objectives),
        constraints=tuple(constraints),
        observables=tuple(observables),
        notes=notes_dict,
    )


def _build_cfg(payload: Mapping[str, Any]) -> StrategyConfig:
    budget_payload = payload.get("budget") or {}
    budget = OptimizationBudget(
        max_iterations=int(budget_payload.get("max_iterations", 25)),
        max_sim_runs=budget_payload.get("max_sim_runs"),
        timeout_s=budget_payload.get("timeout_s"),
        no_improve_patience=int(budget_payload.get("no_improve_patience", 5)),
    )
    notes = payload.get("notes") or {}
    notes_dict = dict(notes) if isinstance(notes, Mapping) else {}
    return StrategyConfig(
        budget=budget,
        seed=payload.get("seed"),
        notes=notes_dict,
    )


def _load_rev_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_spec(store: SessionStore, state: SessionState) -> CircuitSpec:
    path = store.session_dir / "spec_revs" / f"spec_rev{state.spec_rev:04d}.json"
    return _build_spec(_load_rev_payload(path))


def _latest_cfg(store: SessionStore, state: SessionState) -> StrategyConfig:
    path = store.session_dir / "cfg_revs" / f"cfg_rev{state.cfg_rev:04d}.json"
    return _build_cfg(_load_rev_payload(path))


def _phase_input_hash(*, phase_id: str, spec_hash: str, cfg_hash: str, extra: Mapping[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"phase": phase_id, "spec_hash": spec_hash, "cfg_hash": cfg_hash}
    if extra:
        payload["extra"] = dict(extra)
    return hash_payload(payload)


def _phase_run_id(session_id: str, phase_id: str, *, spec_rev: int, cfg_rev: int) -> str:
    return f"{session_id}_{phase_id}_s{spec_rev:04d}_c{cfg_rev:04d}"


class InteractiveSessionStrategy:
    """Interactive (checkpoint/continue) wrapper for baseline->grid->corner_validate runs."""

    name = "interactive_session"
    version = "0.1.0"

    def __init__(
        self,
        *,
        grid_strategy: GridSearchStrategy | None = None,
        baseline_strategy: NoOptBaselineStrategy | None = None,
        corner_validate_op: CornerValidateOperator | None = None,
        measure_fn: Any = None,
    ) -> None:
        self.grid = grid_strategy or GridSearchStrategy(measure_fn=measure_fn)
        self.baseline = baseline_strategy or NoOptBaselineStrategy()
        self.corner_validate = corner_validate_op or CornerValidateOperator(measure_fn=measure_fn)

    def start_session(
        self,
        *,
        bench_id: str,
        source: CircuitSource,
        spec: CircuitSpec,
        cfg: StrategyConfig,
        workspace_root: Path,
        run_to_phase: str = "p2_corner_validate",
        actor: str = "human",
        reason: str = "start_session",
    ) -> RunContext:
        session_ctx = RunContext(workspace_root=workspace_root, seed=cfg.seed)
        store = SessionStore(session_ctx.run_dir())
        store.create_session(
            session_id=session_ctx.run_id,
            bench_id=bench_id,
            seed=cfg.seed,
            spec=spec,
            cfg=cfg,
            actor=actor,
            reason=reason,
        )

        self._run_phases(
            store=store,
            session_ctx=session_ctx,
            source=source,
            spec=spec,
            cfg=cfg,
            run_to_phase=run_to_phase,
        )
        return session_ctx

    def continue_session(
        self,
        *,
        session_run_dir: Path,
        source: CircuitSource,
        spec_delta: Mapping[str, Any] | None = None,
        cfg_delta: Mapping[str, Any] | None = None,
        next_phase: str = "p2_corner_validate",
        actor: str = "human",
        reason: str = "continue_session",
    ) -> None:
        store = SessionStore(Path(session_run_dir))
        state = store.load_session_state()
        spec = _latest_spec(store, state)
        cfg = _latest_cfg(store, state)

        # Apply deltas (pure) and record traces.
        if spec_delta:
            delta = SpecDelta.from_dict(spec_delta)
            new_spec = apply_spec_delta(spec, delta)
            old_hash = state.spec_hash
            new_hash = hash_spec(new_spec)
            if new_hash != old_hash:
                diff = diff_specs(spec, new_spec)
                rev = state.spec_rev + 1
                store.write_spec_revision(rev=rev, spec=new_spec)
                entry = TraceEntry(
                    kind="spec",
                    rev=rev,
                    timestamp=_utc_now_iso(),
                    actor=actor,
                    reason=reason,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    diff=diff,
                    linked_phase=state.current_phase,
                )
                store.append_trace(entry)
                store.update_session_state(
                    lambda s: replace(
                        s,
                        spec_rev=rev,
                        spec_hash=new_hash,
                    )
                )
                state = store.load_session_state()
                spec = new_spec

        if cfg_delta:
            delta = _cfg_delta_from_mapping(cfg_delta) if not isinstance(cfg_delta, CfgDelta) else cfg_delta
            new_cfg = apply_cfg_delta(cfg, delta)
            old_hash = state.cfg_hash
            new_hash = hash_cfg(new_cfg)
            if new_hash != old_hash:
                rev = state.cfg_rev + 1
                store.write_cfg_revision(rev=rev, cfg=new_cfg)
                entry = TraceEntry(
                    kind="cfg",
                    rev=rev,
                    timestamp=_utc_now_iso(),
                    actor=actor,
                    reason=reason,
                    old_hash=old_hash,
                    new_hash=new_hash,
                    diff=_diff_cfg(cfg, new_cfg),
                    linked_phase=state.current_phase,
                )
                store.append_trace(entry)
                store.update_session_state(lambda s: replace(s, cfg_rev=rev, cfg_hash=new_hash))
                state = store.load_session_state()
                cfg = new_cfg

        # Re-run phases if needed based on checkpoints.
        session_ctx = RunContext(workspace_root=store.run_dir.parents[1], run_id=state.session_id, seed=state.seed)
        self._run_phases(
            store=store,
            session_ctx=session_ctx,
            source=source,
            spec=spec,
            cfg=cfg,
            run_to_phase=next_phase,
        )

    def _run_phases(
        self,
        *,
        store: SessionStore,
        session_ctx: RunContext,
        source: CircuitSource,
        spec: CircuitSpec,
        cfg: StrategyConfig,
        run_to_phase: str,
    ) -> None:
        # Phase order is fixed for Step6.
        phases = ["p0_baseline", "p1_grid", "p2_corner_validate"]
        phase_stop = phases.index(run_to_phase) if run_to_phase in phases else len(phases) - 1

        state = store.load_session_state()
        spec_h = hash_spec(spec)
        cfg_h = hash_cfg(cfg)

        # P0 baseline (optional for CI/mock sessions).
        if 0 <= phase_stop:
            baseline_cfg = cfg
            input_hash = _phase_input_hash(phase_id="p0_baseline", spec_hash=spec_h, cfg_hash=cfg_h)
            ck = store.load_checkpoint("p0_baseline")
            if ck and ck.get("input_hash") == input_hash:
                store.record_phase(
                    phase_id="p0_baseline",
                    status="skipped",
                    run_dir=ck.get("run_dir"),
                    spec_hash=spec_h,
                    cfg_hash=cfg_h,
                    input_hash=input_hash,
                    output_summary=ck.get("output_summary") or {},
                    stop_reason=ck.get("stop_reason"),
                    notes={"skip_reason": "unchanged"},
                )
            else:
                # Allow disabling baseline via cfg.notes["session"]["run_baseline"]=false for mock tests.
                session_notes = cfg.notes.get("session") if isinstance(cfg.notes.get("session"), Mapping) else {}
                if session_notes.get("run_baseline") is False:
                    store.write_checkpoint(
                        "p0_baseline",
                        {
                            "phase_id": "p0_baseline",
                            "input_hash": input_hash,
                            "run_dir": None,
                            "stop_reason": None,
                            "output_summary": {"skipped": True, "reason": "run_baseline=false"},
                        },
                    )
                    store.record_phase(
                        phase_id="p0_baseline",
                        status="skipped",
                        run_dir=None,
                        spec_hash=spec_h,
                        cfg_hash=cfg_h,
                        input_hash=input_hash,
                        output_summary={"skipped": True, "reason": "run_baseline=false"},
                        stop_reason=None,
                        notes={"skip_reason": "disabled"},
                    )
                else:
                    ws_root = session_ctx.workspace_root
                    run_id = _phase_run_id(state.session_id, "p0", spec_rev=state.spec_rev, cfg_rev=state.cfg_rev)
                    p0_ctx = RunContext(workspace_root=ws_root, run_id=run_id, seed=cfg.seed)
                    result = self.baseline.run(spec=spec, source=source, ctx=p0_ctx, cfg=baseline_cfg)
                    summary = {
                        "stop_reason": getattr(result.stop_reason, "value", None),
                        "best_score": result.notes.get("best_score"),
                        "all_pass": result.notes.get("all_pass"),
                        "run_id": p0_ctx.run_id,
                    }
                    store.write_checkpoint(
                        "p0_baseline",
                        {
                            "phase_id": "p0_baseline",
                            "input_hash": input_hash,
                            "run_dir": str(p0_ctx.run_dir()),
                            "stop_reason": summary["stop_reason"],
                            "output_summary": summary,
                        },
                    )
                    store.record_phase(
                        phase_id="p0_baseline",
                        status="completed",
                        run_dir=str(p0_ctx.run_dir()),
                        spec_hash=spec_h,
                        cfg_hash=cfg_h,
                        input_hash=input_hash,
                        output_summary=summary,
                        stop_reason=summary["stop_reason"],
                    )

        # Reload after potential updates.
        state = store.load_session_state()

        # P1 grid
        if 1 <= phase_stop:
            input_hash = _phase_input_hash(
                phase_id="p1_grid",
                spec_hash=spec_h,
                cfg_hash=cfg_h,
                extra={"grid_search": dict(cfg.notes.get("grid_search") or {})},
            )
            ck = store.load_checkpoint("p1_grid")
            if ck and ck.get("input_hash") == input_hash:
                grid_run_dir = ck.get("run_dir")
                store.record_phase(
                    phase_id="p1_grid",
                    status="skipped",
                    run_dir=grid_run_dir,
                    spec_hash=spec_h,
                    cfg_hash=cfg_h,
                    input_hash=input_hash,
                    output_summary=ck.get("output_summary") or {},
                    stop_reason=ck.get("stop_reason"),
                    notes={"skip_reason": "unchanged"},
                )
            else:
                ws_root = session_ctx.workspace_root
                run_id = _phase_run_id(state.session_id, "p1", spec_rev=state.spec_rev, cfg_rev=state.cfg_rev)
                p1_ctx = RunContext(workspace_root=ws_root, run_id=run_id, seed=cfg.seed)
                result = self.grid.run(spec=spec, source=source, ctx=p1_ctx, cfg=cfg)
                summary = {
                    "stop_reason": getattr(result.stop_reason, "value", None),
                    "best_score": result.notes.get("best_score"),
                    "all_pass": result.notes.get("all_pass"),
                    "run_id": p1_ctx.run_id,
                }
                store.write_checkpoint(
                    "p1_grid",
                    {
                        "phase_id": "p1_grid",
                        "input_hash": input_hash,
                        "run_dir": str(p1_ctx.run_dir()),
                        "stop_reason": summary["stop_reason"],
                        "output_summary": summary,
                    },
                )
                store.record_phase(
                    phase_id="p1_grid",
                    status="completed",
                    run_dir=str(p1_ctx.run_dir()),
                    spec_hash=spec_h,
                    cfg_hash=cfg_h,
                    input_hash=input_hash,
                    output_summary=summary,
                    stop_reason=summary["stop_reason"],
                )

        state = store.load_session_state()
        grid_ck = store.load_checkpoint("p1_grid") or {}
        grid_run_dir = grid_ck.get("run_dir")

        # P2 corner validate (writes into the grid run_dir).
        if 2 <= phase_stop:
            if not grid_run_dir:
                raise ValidationError("corner_validate requires a completed p1_grid run_dir")
            input_hash = _phase_input_hash(
                phase_id="p2_corner_validate",
                spec_hash=spec_h,
                cfg_hash=cfg_h,
                extra={"grid_run_dir": grid_run_dir, "corner_validate": dict(cfg.notes.get("corner_validate") or {})},
            )
            ck = store.load_checkpoint("p2_corner_validate")
            if ck and ck.get("input_hash") == input_hash:
                store.record_phase(
                    phase_id="p2_corner_validate",
                    status="skipped",
                    run_dir=grid_run_dir,
                    spec_hash=spec_h,
                    cfg_hash=cfg_h,
                    input_hash=input_hash,
                    output_summary=ck.get("output_summary") or {},
                    stop_reason=ck.get("stop_reason"),
                    notes={"skip_reason": "unchanged"},
                )
            else:
                run_dir_path = Path(grid_run_dir)
                recorder = RunRecorder(run_dir_path)
                # Best-effort manifest load: CornerValidateOperator expects a RunManifest-like object.
                manifest = None
                manifest_path = run_dir_path / "run_manifest.json"
                if manifest_path.exists():
                    try:
                        from ..contracts.provenance import RunManifest

                        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if isinstance(payload, Mapping):
                            manifest = RunManifest(
                                run_id=str(payload.get("run_id") or run_dir_path.name),
                                workspace=Path(payload.get("workspace") or run_dir_path),
                                seed=payload.get("seed"),
                                timestamp_start=payload.get("timestamp_start"),
                                timestamp_end=payload.get("timestamp_end"),
                                inputs=dict(payload.get("inputs") or {}),
                                environment=dict(payload.get("environment") or {}),
                                files=dict(payload.get("files") or {}),
                                result_summary=dict(payload.get("result_summary") or {}),
                                tool_versions=dict(payload.get("tool_versions") or {}),
                                env=dict(payload.get("env") or {}),
                                notes=dict(payload.get("notes") or {}),
                            )
                    except Exception:
                        manifest = None

                self.corner_validate.run(
                    {
                        "run_dir": run_dir_path,
                        "recorder": recorder,
                        "manifest": manifest,
                        "source": source,
                        "spec": spec,
                        "cfg": cfg,
                        "candidates_source": cfg.notes.get("corner_validate", {}).get("candidates_source", "topk")
                        if isinstance(cfg.notes.get("corner_validate"), Mapping)
                        else "topk",
                        "corner_validate": cfg.notes.get("corner_validate") or {},
                    },
                    ctx=session_ctx,
                )

                robust_topk_path = run_dir_path / "search" / "robust_topk.json"
                robust = []
                if robust_topk_path.exists():
                    robust = json.loads(robust_topk_path.read_text(encoding="utf-8"))
                summary = {"robust_topk_count": len(robust) if isinstance(robust, list) else 0}
                store.write_checkpoint(
                    "p2_corner_validate",
                    {
                        "phase_id": "p2_corner_validate",
                        "input_hash": input_hash,
                        "run_dir": grid_run_dir,
                        "stop_reason": None,
                        "output_summary": summary,
                    },
                )
                store.record_phase(
                    phase_id="p2_corner_validate",
                    status="completed",
                    run_dir=grid_run_dir,
                    spec_hash=spec_h,
                    cfg_hash=cfg_h,
                    input_hash=input_hash,
                    output_summary=summary,
                    stop_reason=None,
                )

        # Always refresh meta report after phase actions.
        report = build_meta_report(store)
        store.recorder.write_text("session/meta_report.md", report)
