from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import copy

from ..agents import CircuitAnalysisAgent, KnobAgent, SearchPlannerAgent, SpecSynthAgent
from ..agents.base import AgentContext
from ..contracts import CircuitSource, CircuitSpec, MetricsBundle, MetricValue, RunResult, StrategyConfig
from ..contracts.enums import SourceKind, StopReason
from ..contracts.errors import ValidationError
from ..contracts.plan import Action
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..operators.netlist import TopologySignatureOperator
from ..runtime.artifact_store import ArtifactStore
from ..runtime.context import RunContext
from ..runtime.plan_executor import PlanExecutor
from ..runtime.recorder import RunRecorder
from ..runtime.recording_utils import finalize_run, record_operator_result
from ..runtime.run_loader import RunLoader
from ..runtime.tool_registry import ToolRegistry
from .corner_search import CornerSearchStrategy
from .grid_search import GridSearchStrategy


def _merge_notes(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            nested = dict(out[k])
            nested.update(dict(v))
            out[k] = nested
        else:
            out[k] = v
    return out


def _metrics_bundle_from_dict(payload: Mapping[str, Any]) -> MetricsBundle:
    mb = MetricsBundle()
    for name, entry in payload.items():
        if not isinstance(entry, Mapping):
            continue
        mv = MetricValue(
            name=str(name),
            value=entry.get("value"),
            unit=str(entry.get("unit") or ""),
            passed=entry.get("passed"),
            details=dict(entry.get("details") or {}),
        )
        mb.values[str(name)] = mv
    return mb


def _load_best_from_run_dir(run_dir: Path, *, source: CircuitSource) -> tuple[Optional[CircuitSource], MetricsBundle]:
    payload = RunLoader(run_dir).load_best()
    best_sp = payload.get("best_sp", "")
    best_metrics = payload.get("best_metrics", {})

    best_source: Optional[CircuitSource] = None
    if best_sp:
        best_source = CircuitSource(kind=source.kind, text=best_sp, name=source.name, metadata=dict(source.metadata))
    if not isinstance(best_metrics, Mapping):
        return best_source, MetricsBundle()
    return best_source, _metrics_bundle_from_dict(best_metrics)


@dataclass
class MultiAgentOrchestratorStrategy:
    """Sequential multi-agent orchestrator.

    Keeps the current project philosophy:
    - Agents only propose structured artifacts/configs (no IO).
    - Operators/Strategies do deterministic execution and recording.
    """

    name: str = "multi_agent_orchestrator"
    version: str = "0.1.0"

    signature_op: Any = None
    circuit_agent: Any = None
    knob_agent: Any = None
    spec_agent: Any = None
    search_agent: Any = None

    # forwarded into search strategies for testing (bypass ngspice)
    measure_fn: Any = None

    def __post_init__(self) -> None:
        if self.signature_op is None:
            self.signature_op = TopologySignatureOperator()
        if self.circuit_agent is None:
            self.circuit_agent = CircuitAnalysisAgent()
        if self.knob_agent is None:
            self.knob_agent = KnobAgent()
        if self.spec_agent is None:
            self.spec_agent = SpecSynthAgent()
        if self.search_agent is None:
            self.search_agent = SearchPlannerAgent()

    def run(
        self,
        spec: CircuitSpec,
        source: CircuitSource,
        ctx: Any,
        cfg: StrategyConfig,
    ) -> RunResult:
        # Recorder/manifest are optional.
        recorder: RunRecorder | None = None
        manifest = None
        if hasattr(ctx, "recorder") and hasattr(ctx, "manifest"):
            try:
                recorder = ctx.recorder()
                manifest = ctx.manifest()
            except Exception:
                recorder = None
                manifest = None

        # --- Stage 0: Parse + signature (deterministic operator) ---
        sig_res = self.signature_op.run(
            {
                "netlist_text": source.text,
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
            },
            ctx=None,
        )
        record_operator_result(recorder, sig_res)
        outs = sig_res.outputs
        circuit_ir = outs["circuit_ir"]
        signature = outs["signature"]
        signature_result = outs.get("signature_result")
        raw_source = source

        if signature_result is not None:
            sanitized_text = signature_result.sanitize_result.sanitized_text
            if sanitized_text != raw_source.text:
                source = CircuitSource(
                    kind=raw_source.kind,
                    text=sanitized_text,
                    name=raw_source.name,
                    metadata=dict(raw_source.metadata),
                )

        # --- Orchestrator store ---
        store: ArtifactStore | None = None
        if recorder is not None:
            store = ArtifactStore(recorder)
            store.put("source", source, producer="orchestrator")
            store.put("signature", signature, producer="orchestrator", kind="text")
            # circuit_ir can be large; keep it as JSON for replay.
            store.put("circuit_ir", circuit_ir, producer="orchestrator")

        # --- Agents (pure) ---
        agent_notes: list[dict[str, Any]] = []

        def _run_agent(agent: Any, *, inputs: Mapping[str, Any]) -> Dict[str, Any]:
            a_ctx = AgentContext(source=source, circuit_ir=circuit_ir, signature=signature, cfg=cfg, spec=spec)
            out = agent.run(a_ctx, inputs)
            if not isinstance(out, dict):
                raise ValidationError(f"agent '{getattr(agent, 'name', type(agent).__name__)}' must return a dict")
            return out

        # 1) Circuit analysis
        circuit_out = _run_agent(self.circuit_agent, inputs={})
        agent_notes.append({"agent": self.circuit_agent.name, "outputs": list(circuit_out.keys())})
        if store is not None:
            for k, v in circuit_out.items():
                store.put(f"agents/{self.circuit_agent.name}/{k}", v, producer=self.circuit_agent.name)

        # 2) Knobs / param rules
        knobs_out = _run_agent(self.knob_agent, inputs=circuit_out)
        agent_notes.append({"agent": self.knob_agent.name, "outputs": list(knobs_out.keys())})
        if store is not None:
            for k, v in knobs_out.items():
                store.put(f"agents/{self.knob_agent.name}/{k}", v, producer=self.knob_agent.name)

        # 3) Spec synthesis
        spec_out = _run_agent(self.spec_agent, inputs={**circuit_out, **knobs_out})
        agent_notes.append({"agent": self.spec_agent.name, "outputs": list(spec_out.keys())})
        if store is not None:
            for k, v in spec_out.items():
                store.put(f"agents/{self.spec_agent.name}/{k}", v, producer=self.spec_agent.name)

        # 4) Search planning
        search_out = _run_agent(self.search_agent, inputs={**circuit_out, **knobs_out, **spec_out})
        agent_notes.append({"agent": self.search_agent.name, "outputs": list(search_out.keys())})
        if store is not None:
            for k, v in search_out.items():
                store.put(f"agents/{self.search_agent.name}/{k}", v, producer=self.search_agent.name)

        selected_spec = spec_out.get("spec")
        if not isinstance(selected_spec, CircuitSpec):
            raise ValidationError("SpecSynthAgent did not produce a CircuitSpec")

        param_rules = knobs_out.get("param_rules")
        if not isinstance(param_rules, Mapping):
            raise ValidationError("KnobAgent did not produce param_rules")

        cfg_notes_update = dict(search_out.get("cfg_notes") or {})
        strategy_choice = str(search_out.get("strategy") or "grid_search")

        # --- Build child StrategyConfig ---
        child_cfg = StrategyConfig(
            budget=cfg.budget,
            seed=cfg.seed,
            notes=_merge_notes(cfg.notes or {}, {"param_rules": dict(param_rules), **cfg_notes_update}),
        )

        # --- Plan: one deterministic search stage ---
        plan: tuple[Action, ...]
        if strategy_choice == "corner_search":
            plan = (Action(op="run_corner_search", inputs=("spec", "source", "cfg"), outputs=("search_run",), params={}),)
        else:
            plan = (Action(op="run_grid_search", inputs=("spec", "source", "cfg"), outputs=("search_run",), params={}),)

        if store is not None:
            store.put("spec", selected_spec, producer="orchestrator")
            store.put("cfg", child_cfg, producer="orchestrator")
            store.put("plan", [a.__dict__ for a in plan], producer="orchestrator")

        if recorder is not None:
            recorder.write_json("orchestrator/agents.json", {"agents": agent_notes})
            if store is not None:
                store.dump_index()
            recorder.write_json("orchestrator/plan.json", [a.__dict__ for a in plan])

        # --- Tool registry: stage runners (deterministic) ---
        registry = ToolRegistry()

        def _make_child_ctx(parent_ctx: Any, run_id: str) -> RunContext:
            if isinstance(parent_ctx, RunContext):
                return RunContext(workspace_root=parent_ctx.workspace_root, run_id=run_id, seed=parent_ctx.seed)
            # Fallback: minimal; assume ctx has workspace_root.
            ws = getattr(parent_ctx, "workspace_root", Path.cwd())
            return RunContext(workspace_root=ws, run_id=run_id)

        def _run_grid(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
            s = GridSearchStrategy(measure_fn=self.measure_fn)
            action_idx = int(params.get("_action_idx", 0))
            child_run_id = f"{getattr(ctx, 'run_id', 'run')}_a{action_idx:02d}_grid"
            child_ctx = _make_child_ctx(ctx, child_run_id)
            result = s.run(spec=inputs["spec"], source=inputs["source"], ctx=child_ctx, cfg=inputs["cfg"])  # type: ignore[arg-type]
            return {
                "search_run": {
                    "run_id": child_ctx.run_id,
                    "run_dir": str(child_ctx.run_dir()),
                    "strategy": s.name,
                    "strategy_version": s.version,
                    "stop_reason": getattr(result.stop_reason, "value", None),
                    "best_score": result.notes.get("best_score"),
                }
            }

        def _run_corner(inputs: Mapping[str, Any], _ctx: Any, params: Mapping[str, Any]) -> Mapping[str, Any]:
            s = CornerSearchStrategy(measure_fn=self.measure_fn)
            action_idx = int(params.get("_action_idx", 0))
            child_run_id = f"{getattr(ctx, 'run_id', 'run')}_a{action_idx:02d}_corner"
            child_ctx = _make_child_ctx(ctx, child_run_id)
            result = s.run(spec=inputs["spec"], source=inputs["source"], ctx=child_ctx, cfg=inputs["cfg"])  # type: ignore[arg-type]
            return {
                "search_run": {
                    "run_id": child_ctx.run_id,
                    "run_dir": str(child_ctx.run_dir()),
                    "strategy": s.name,
                    "strategy_version": s.version,
                    "stop_reason": getattr(result.stop_reason, "value", None),
                    "best_score": result.notes.get("best_score"),
                }
            }

        registry.register("run_grid_search", _run_grid)
        registry.register("run_corner_search", _run_corner)

        executor = PlanExecutor(registry)
        if store is None:
            # No recorder -> still run but with an ephemeral store.
            class _DummyRecorder:
                def __init__(self) -> None:
                    self.run_dir = Path.cwd()
                def write_json(self, *_a: Any, **_k: Any) -> None:
                    return None
                def write_text(self, *_a: Any, **_k: Any) -> None:
                    return None
                def append_jsonl(self, *_a: Any, **_k: Any) -> None:
                    return None
            dummy = _DummyRecorder()
            store = ArtifactStore(dummy)  # type: ignore[arg-type]
            store.put("spec", selected_spec)
            store.put("source", source)
            store.put("cfg", child_cfg)

        events = executor.execute(plan, store=store, ctx=None, recorder=recorder)

        search_info = store.get("search_run")
        if not isinstance(search_info, Mapping):
            raise ValidationError("search stage did not produce a valid search_run")
        run_dir = Path(str(search_info.get("run_dir"))).resolve()

        best_source, best_metrics = _load_best_from_run_dir(run_dir, source=source)
        stop_reason = None
        stop_raw = search_info.get("stop_reason")
        if isinstance(stop_raw, str):
            try:
                stop_reason = StopReason(stop_raw)
            except Exception:
                stop_reason = StopReason.error

        # --- Orchestrator report ---
        if recorder is not None:
            lines: list[str] = []
            lines.append("# Multi-Agent Orchestrator Report")
            lines.append("")
            lines.append("## Summary")
            lines.append(f"- signature: {signature}")
            lines.append(f"- selected_search_strategy: {search_info.get('strategy')}")
            lines.append(f"- search_run_dir: {search_info.get('run_dir')}")
            lines.append("")
            lines.append("## Agents")
            for entry in agent_notes:
                lines.append(f"- {entry.get('agent')}: outputs={entry.get('outputs')}")
            lines.append("")
            lines.append("## Plan")
            for a in plan:
                lines.append(f"- op: {a.op} inputs={list(a.inputs)} outputs={list(a.outputs)}")
            lines.append("")
            lines.append("## Search Result")
            lines.append(f"- stop_reason: {search_info.get('stop_reason')}")
            lines.append(f"- best_score: {search_info.get('best_score')}")
            for name, mv in best_metrics.values.items():
                lines.append(f"- metric {name}: {mv.value} {mv.unit or ''}".rstrip())
            recorder.write_text("report.md", "\n".join(lines))

        # --- Manifest: record orchestrator inputs ---
        if manifest is not None:
            manifest.environment.setdefault("strategy_name", self.name)
            manifest.environment.setdefault("strategy_version", self.version)
            manifest.inputs.update(
                {
                    "netlist_sha256": stable_hash_str(source.text),
                    "spec_sha256": stable_hash_json({"objectives": [o.__dict__ for o in selected_spec.objectives]}),
                    "cfg_sha256": stable_hash_json({"notes": child_cfg.notes, "seed": child_cfg.seed}),
                    "signature": signature,
                    "child_run_dir": str(run_dir),
                }
            )

        recording_errors = finalize_run(
            recorder=recorder,
            manifest=manifest,
            best_source=best_source,
            best_metrics=best_metrics,
            history=[],
            stop_reason=stop_reason,
            best_score=float(search_info.get("best_score") or float("inf")),
            best_iter=None,
            sim_runs=0,
            sim_runs_ok=0,
            sim_runs_failed=0,
        )

        notes = {
            "child_run": dict(search_info),
            "agent_modes": {
                "knobs": knobs_out.get("mode"),
                "spec": spec_out.get("mode"),
                "search": search_out.get("mode"),
            },
            "recording_errors": recording_errors,
            "plan_events": [e.to_dict() for e in events],
        }
        return RunResult(
            best_source=best_source,
            best_metrics=best_metrics,
            history=[],
            stop_reason=stop_reason,
            notes=notes,
        )
