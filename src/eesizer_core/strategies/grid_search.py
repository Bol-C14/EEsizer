from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..analysis.pareto import objective_losses, pareto_front, top_k
from ..contracts import (
    CircuitSource,
    CircuitSpec,
    MetricsBundle,
    Patch,
    PatchOp,
    RunResult,
    StrategyConfig,
)
from ..contracts.errors import ValidationError
from ..contracts.enums import PatchOpType, StopReason
from ..contracts.strategy import Strategy
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ..domain.spice.patching import extract_param_values
from ..metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ..operators.guards import (
    BehaviorGuardOperator,
    GuardChainOperator,
    PatchGuardOperator,
    TopologyGuardOperator,
)
from ..operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ..runtime.recorder import RunRecorder
from ..runtime.recording_utils import (
    attempt_record,
    finalize_run,
    guard_report_to_dict,
    param_space_to_dict,
    patch_to_dict,
    record_history_entry,
    record_operator_result,
    spec_to_dict,
    strategy_cfg_to_dict,
)
from ..search.samplers import coordinate_candidates, factorial_candidates, make_levels
from ..sim import DeckBuildOperator, NgspiceRunOperator
from .attempt_pipeline import AttemptOperators, run_attempt
from .patch_loop.evaluate import MeasureFn, evaluate_metrics, run_baseline
from .patch_loop.planning import group_metric_names_by_kind


def _grid_cfg(notes: Mapping[str, Any]) -> dict[str, Any]:
    raw = notes.get("grid_search")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


@dataclass
class _GridPrepared:
    history: list[dict[str, Any]]
    recorder: RunRecorder | None
    manifest: Any
    cfg: StrategyConfig
    source: CircuitSource
    circuit_ir: Any
    signature: str
    param_space: Any
    guard_cfg: dict[str, Any]
    grid_cfg: dict[str, Any]
    metric_groups: Mapping[Any, list[str]]
    max_iters: int
    max_sim_runs: int | None
    candidate_budget: int
    mode: str
    levels: int
    span_mul: float
    scale: str
    top_k_count: int
    stop_on_first_pass: bool
    baseline_retries: int


@dataclass
class _GridState:
    best_source: CircuitSource
    best_metrics: MetricsBundle
    best_score: float
    best_all_pass: bool
    best_iter: int | None
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int


@dataclass
class _GridBaselineOutcome:
    result: RunResult | None
    baseline: Any | None
    eval0: dict[str, Any] | None
    state: _GridState | None
    stop_reason: StopReason | None


@dataclass
class _GridCandidates:
    candidates: list[dict[str, float]]
    param_value_errors: list[str]


@dataclass
class GridSearchStrategy(Strategy):
    """Deterministic grid/coordinate search strategy."""

    name: str = "grid_search"
    version: str = "0.1.0"

    signature_op: Any = None
    patch_apply_op: Any = None
    patch_guard_op: Any = None
    topology_guard_op: Any = None
    behavior_guard_op: Any = None
    guard_chain_op: Any = None
    deck_build_op: Any = None
    sim_run_op: Any = None
    metrics_op: Any = None
    registry: MetricRegistry | None = None
    measure_fn: MeasureFn | None = None

    def __post_init__(self) -> None:
        if self.signature_op is None:
            self.signature_op = TopologySignatureOperator()
        if self.patch_apply_op is None:
            self.patch_apply_op = PatchApplyOperator()
        if self.patch_guard_op is None:
            self.patch_guard_op = PatchGuardOperator()
        if self.topology_guard_op is None:
            self.topology_guard_op = TopologyGuardOperator()
        if self.behavior_guard_op is None:
            self.behavior_guard_op = BehaviorGuardOperator()
        if self.guard_chain_op is None:
            self.guard_chain_op = GuardChainOperator()
        if self.deck_build_op is None:
            self.deck_build_op = DeckBuildOperator()
        if self.sim_run_op is None:
            self.sim_run_op = NgspiceRunOperator()
        if self.metrics_op is None:
            self.metrics_op = ComputeMetricsOperator()
        if self.registry is None:
            self.registry = DEFAULT_REGISTRY

    def _prepare(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> _GridPrepared:
        history: list[dict[str, Any]] = []
        recorder: RunRecorder | None = None
        manifest = None
        if hasattr(ctx, "recorder") and hasattr(ctx, "manifest"):
            try:
                recorder = ctx.recorder()
                manifest = ctx.manifest()
            except Exception:
                recorder = None
                manifest = None

        sig_result = self.signature_op.run(
            {
                "netlist_text": source.text,
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
            },
            ctx=None,
        )
        record_operator_result(recorder, sig_result)
        sig_res = sig_result.outputs
        circuit_ir = sig_res["circuit_ir"]
        signature = sig_res["signature"]
        signature_result = sig_res.get("signature_result")
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

        rules = ParamInferenceRules(**cfg.notes.get("param_rules", {}))
        param_space = infer_param_space_from_ir(
            circuit_ir,
            rules=rules,
            frozen_param_ids=cfg.notes.get("frozen_param_ids", ()),
        )
        guard_cfg = dict(cfg.notes.get("guard_cfg", {}))
        guard_cfg.setdefault("wl_ratio_min", rules.wl_ratio_min)
        guard_cfg.setdefault("max_mul_factor", cfg.notes.get("max_mul_factor", 10.0))
        guard_cfg.setdefault("max_patch_ops", cfg.notes.get("max_patch_ops", 20))
        if "max_add_delta" in cfg.notes and "max_add_delta" not in guard_cfg:
            guard_cfg["max_add_delta"] = cfg.notes.get("max_add_delta")

        if manifest is not None:
            manifest.environment.setdefault("strategy_name", self.name)
            manifest.environment.setdefault("strategy_version", self.version)
            spec_payload = spec_to_dict(spec)
            param_payload = param_space_to_dict(param_space)
            cfg_payload = strategy_cfg_to_dict(cfg, guard_cfg)
            manifest.inputs.update(
                {
                    "netlist_sha256": stable_hash_str(source.text),
                    "spec_sha256": stable_hash_json(spec_payload),
                    "param_space_sha256": stable_hash_json(param_payload),
                    "cfg_sha256": stable_hash_json(cfg_payload),
                    "signature": signature,
                }
            )
            manifest.files.setdefault("inputs/source.sp", "inputs/source.sp")
            manifest.files.setdefault("inputs/spec.json", "inputs/spec.json")
            manifest.files.setdefault("inputs/param_space.json", "inputs/param_space.json")
            manifest.files.setdefault("inputs/cfg.json", "inputs/cfg.json")
            manifest.files.setdefault("inputs/signature.txt", "inputs/signature.txt")
            manifest.files.setdefault("history/iterations.jsonl", "history/iterations.jsonl")
            manifest.files.setdefault("history/summary.json", "history/summary.json")
            manifest.files.setdefault("provenance/operator_calls.jsonl", "provenance/operator_calls.jsonl")
            manifest.files.setdefault("best/best.sp", "best/best.sp")
            manifest.files.setdefault("best/best_metrics.json", "best/best_metrics.json")
            manifest.files.setdefault("search/candidates.json", "search/candidates.json")
            manifest.files.setdefault("search/topk.json", "search/topk.json")
            manifest.files.setdefault("search/pareto.json", "search/pareto.json")
            manifest.files.setdefault("report.md", "report.md")
            if recorder is not None:
                recorder.write_input("source.sp", source.text)
                recorder.write_input("spec.json", spec_payload)
                recorder.write_input("param_space.json", param_payload)
                recorder.write_input("cfg.json", cfg_payload)
                recorder.write_input("signature.txt", signature)

        grid_cfg = _grid_cfg(cfg.notes)
        mode = str(grid_cfg.get("mode", "coordinate")).lower()
        levels = int(grid_cfg.get("levels", 10))
        span_mul = float(grid_cfg.get("span_mul", 10.0))
        scale = str(grid_cfg.get("scale", "log")).lower()
        top_k_count = int(grid_cfg.get("top_k", 5))
        stop_on_first_pass = bool(grid_cfg.get("stop_on_first_pass", False))
        baseline_retries = int(grid_cfg.get("baseline_retries", 0))

        max_iters = cfg.budget.max_iterations
        max_sim_runs = cfg.budget.max_sim_runs
        candidate_budget = max(0, max_iters - 1)

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)

        return _GridPrepared(
            history=history,
            recorder=recorder,
            manifest=manifest,
            cfg=cfg,
            source=source,
            circuit_ir=circuit_ir,
            signature=signature,
            param_space=param_space,
            guard_cfg=guard_cfg,
            grid_cfg=grid_cfg,
            metric_groups=metric_groups,
            max_iters=max_iters,
            max_sim_runs=max_sim_runs,
            candidate_budget=candidate_budget,
            mode=mode,
            levels=levels,
            span_mul=span_mul,
            scale=scale,
            top_k_count=top_k_count,
            stop_on_first_pass=stop_on_first_pass,
            baseline_retries=baseline_retries,
        )

    def _baseline_phase(
        self,
        prepared: _GridPrepared,
        spec: CircuitSpec,
        ctx: Any,
    ) -> _GridBaselineOutcome:
        baseline = run_baseline(
            source=prepared.source,
            spec=spec,
            metric_groups=prepared.metric_groups,
            ctx=ctx,
            guard_cfg=prepared.guard_cfg,
            max_retries=prepared.baseline_retries,
            max_sim_runs=prepared.max_sim_runs,
            recorder=prepared.recorder,
            manifest=prepared.manifest,
            measure_fn=self.measure_fn,
            deck_build_op=self.deck_build_op,
            sim_run_op=self.sim_run_op,
            metrics_op=self.metrics_op,
            behavior_guard_op=self.behavior_guard_op,
            guard_chain_op=self.guard_chain_op,
        )

        if not baseline.success:
            prepared.history.append(
                {
                    "iteration": 0,
                    "candidate": None,
                    "patch": None,
                    "signature_before": prepared.signature,
                    "signature_after": prepared.signature,
                    "metrics": {},
                    "score": float("inf"),
                    "all_pass": False,
                    "improved": False,
                    "objectives": [],
                    "sim_stages": baseline.stage_map,
                    "warnings": baseline.warnings,
                    "errors": baseline.errors,
                    "guard": guard_report_to_dict(baseline.guard_report) if baseline.guard_report else None,
                    "attempts": baseline.attempts,
                }
            )
            record_history_entry(prepared.recorder, prepared.history[-1])
            recording_errors = finalize_run(
                recorder=prepared.recorder,
                manifest=prepared.manifest,
                best_source=prepared.source,
                best_metrics=MetricsBundle(),
                history=prepared.history,
                stop_reason=baseline.stop_reason,
                best_score=float("inf"),
                best_iter=None,
                sim_runs=baseline.sim_runs,
                sim_runs_ok=baseline.sim_runs_ok,
                sim_runs_failed=baseline.sim_runs_failed,
            )
            return _GridBaselineOutcome(
                result=RunResult(
                    best_source=prepared.source,
                    best_metrics=MetricsBundle(),
                    history=prepared.history,
                    stop_reason=baseline.stop_reason,
                    notes={
                        "best_score": float("inf"),
                        "all_pass": False,
                        "recording_errors": recording_errors,
                    },
                ),
                baseline=None,
                eval0=None,
                state=None,
                stop_reason=baseline.stop_reason,
            )

        eval0 = evaluate_metrics(spec, baseline.metrics)
        state = _GridState(
            best_source=prepared.source,
            best_metrics=baseline.metrics,
            best_score=eval0["score"],
            best_all_pass=eval0["all_pass"],
            best_iter=0,
            sim_runs=baseline.sim_runs,
            sim_runs_ok=baseline.sim_runs_ok,
            sim_runs_failed=baseline.sim_runs_failed,
        )
        stop_reason: StopReason | None = StopReason.reached_target if state.best_all_pass else None

        prepared.history.append(
            {
                "iteration": 0,
                "candidate": None,
                "patch": None,
                "signature_before": prepared.signature,
                "signature_after": prepared.signature,
                "metrics": {k: v.value for k, v in baseline.metrics.values.items()},
                "score": state.best_score,
                "all_pass": state.best_all_pass,
                "improved": False,
                "objectives": eval0["per_objective"],
                "sim_stages": baseline.stage_map,
                "warnings": baseline.warnings,
                "errors": baseline.errors,
                "guard": guard_report_to_dict(baseline.guard_report) if baseline.guard_report else None,
                "attempts": baseline.attempts,
            }
        )
        record_history_entry(prepared.recorder, prepared.history[-1])

        return _GridBaselineOutcome(
            result=None,
            baseline=baseline,
            eval0=eval0,
            state=state,
            stop_reason=stop_reason,
        )

    def _generate_candidates(self, prepared: _GridPrepared) -> _GridCandidates:
        param_ids = [p.param_id for p in prepared.param_space.params if not p.frozen]
        cfg_param_ids = prepared.grid_cfg.get("param_ids")
        if isinstance(cfg_param_ids, (list, tuple)):
            param_ids = [str(pid).lower() for pid in cfg_param_ids]
        allow_param_override = bool(prepared.grid_cfg.get("allow_param_ids_override_frozen", False))
        frozen_conflicts = [
            pid
            for pid in param_ids
            if (prepared.param_space.get(pid) is not None and prepared.param_space.get(pid).frozen)
        ]
        if frozen_conflicts and not allow_param_override:
            raise ValidationError(
                "grid_search param_ids include frozen params: "
                f"{', '.join(sorted(set(frozen_conflicts)))}; set allow_param_ids_override_frozen=true to override"
            )

        baseline_values, param_value_errors = extract_param_values(prepared.circuit_ir, param_ids=param_ids)
        per_param_levels: dict[str, list[float]] = {}
        for param_id in param_ids:
            if param_id not in baseline_values:
                continue
            param_def = prepared.param_space.get(param_id)
            if param_def is None:
                continue
            per_param_levels[param_id] = make_levels(
                nominal=baseline_values[param_id],
                lower=param_def.lower,
                upper=param_def.upper,
                levels=prepared.levels,
                span_mul=prepared.span_mul,
                scale=prepared.scale,
            )

        if prepared.mode == "factorial":
            candidates = factorial_candidates(param_ids, per_param_levels, baseline_values)
        else:
            candidates = coordinate_candidates(param_ids, per_param_levels, baseline_values)

        if prepared.candidate_budget and len(candidates) > prepared.candidate_budget:
            candidates = candidates[: prepared.candidate_budget]

        if prepared.recorder is not None:
            prepared.recorder.write_json("search/candidates.json", candidates)

        return _GridCandidates(candidates=candidates, param_value_errors=param_value_errors)

    def _evaluate_candidates_loop(
        self,
        prepared: _GridPrepared,
        spec: CircuitSpec,
        ctx: Any,
        state: _GridState,
        candidates: list[dict[str, float]],
    ) -> tuple[list[dict[str, Any]], list[list[float]], StopReason, _GridState]:
        validation_opts = {
            "wl_ratio_min": prepared.guard_cfg.get("wl_ratio_min"),
            "max_mul_factor": prepared.guard_cfg.get("max_mul_factor", 10.0),
        }
        apply_opts = {
            "include_paths": prepared.cfg.notes.get("include_paths", True),
            "max_lines": prepared.cfg.notes.get("max_lines", 50000),
            "validation_opts": validation_opts,
        }

        candidate_entries: list[dict[str, Any]] = []
        candidate_losses: list[list[float]] = []
        attempt_ops = AttemptOperators(
            patch_guard_op=self.patch_guard_op,
            patch_apply_op=self.patch_apply_op,
            topology_guard_op=self.topology_guard_op,
            behavior_guard_op=self.behavior_guard_op,
            guard_chain_op=self.guard_chain_op,
            formal_guard_op=None,
            deck_build_op=self.deck_build_op,
            sim_run_op=self.sim_run_op,
            metrics_op=self.metrics_op,
        )

        stop_reason: StopReason | None = None

        for idx, candidate in enumerate(candidates, start=1):
            if prepared.max_sim_runs is not None and state.sim_runs >= prepared.max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            patch_ops = [
                PatchOp(param=pid, op=PatchOpType.set, value=val, why="grid_search")
                for pid, val in candidate.items()
            ]
            patch = Patch(ops=tuple(patch_ops))
            signature_before = prepared.signature
            attempt_result = run_attempt(
                iter_idx=idx,
                attempt=0,
                patch=patch,
                cur_source=prepared.source,
                cur_signature=prepared.signature,
                circuit_ir=prepared.circuit_ir,
                param_space=prepared.param_space,
                spec=spec,
                guard_cfg=prepared.guard_cfg,
                apply_opts=apply_opts,
                metric_groups=prepared.metric_groups,
                ctx=ctx,
                recorder=prepared.recorder,
                manifest=prepared.manifest,
                measure_fn=self.measure_fn,
                ops=attempt_ops,
            )
            state.sim_runs += attempt_result.sim_runs
            state.sim_runs_ok += attempt_result.sim_runs_ok
            state.sim_runs_failed += attempt_result.sim_runs_failed
            metrics_i = attempt_result.metrics
            stage_map_i = attempt_result.stage_map
            warnings_i = attempt_result.warnings
            guard_report = attempt_result.guard_report
            errors = attempt_result.guard_failures
            attempts = [attempt_record(0, patch, guard_report, stage_map_i, warnings_i)]
            new_source = attempt_result.new_source
            new_signature = attempt_result.new_signature
            new_circuit_ir = attempt_result.new_circuit_ir

            if not attempt_result.success or metrics_i is None:
                metrics_payload = {k: v.value for k, v in metrics_i.values.items()} if metrics_i else {}
                prepared.history.append(
                    {
                        "iteration": idx,
                        "candidate": dict(candidate),
                        "patch": patch_to_dict(patch),
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": metrics_payload,
                        "score": float("inf"),
                        "all_pass": False,
                        "improved": False,
                        "objectives": [],
                        "sim_stages": stage_map_i,
                        "warnings": warnings_i,
                        "errors": errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                    }
                )
                record_history_entry(prepared.recorder, prepared.history[-1])
                continue

            eval_i = evaluate_metrics(spec, metrics_i)
            improved = eval_i["score"] < state.best_score - 1e-12
            if improved:
                state.best_score = eval_i["score"]
                state.best_metrics = metrics_i
                state.best_source = new_source
                state.best_all_pass = eval_i["all_pass"]
                state.best_iter = idx

            prepared.history.append(
                {
                    "iteration": idx,
                    "candidate": dict(candidate),
                    "patch": patch_to_dict(patch),
                    "signature_before": signature_before,
                    "signature_after": new_signature,
                    "metrics": {k: v.value for k, v in metrics_i.values.items()},
                    "score": eval_i["score"],
                    "all_pass": eval_i["all_pass"],
                    "improved": improved,
                    "objectives": eval_i["per_objective"],
                    "sim_stages": stage_map_i,
                    "warnings": warnings_i,
                    "errors": errors,
                    "guard": guard_report_to_dict(guard_report) if guard_report else None,
                    "attempts": attempts,
                }
            )
            record_history_entry(prepared.recorder, prepared.history[-1])

            losses = objective_losses(eval_i)
            candidate_entry = {
                "iteration": idx,
                "candidate": dict(candidate),
                "score": eval_i["score"],
                "all_pass": eval_i["all_pass"],
                "losses": list(losses),
                "metrics": {k: v.value for k, v in metrics_i.values.items()},
            }
            candidate_entries.append(candidate_entry)
            candidate_losses.append(losses)

            if prepared.stop_on_first_pass and eval_i["all_pass"]:
                stop_reason = StopReason.reached_target
                break

        if stop_reason is None:
            stop_reason = StopReason.max_iterations

        return candidate_entries, candidate_losses, stop_reason, state

    def _write_report_and_finalize(
        self,
        prepared: _GridPrepared,
        state: _GridState,
        stop_reason: StopReason | None,
        baseline_eval: dict[str, Any],
        baseline_metrics: MetricsBundle,
        candidate_entries: list[dict[str, Any]],
        candidate_losses: list[list[float]],
        param_value_errors: list[str],
    ) -> RunResult:
        if prepared.recorder is not None:
            top_entries = top_k(candidate_entries, prepared.top_k_count)
            pareto_idx = pareto_front(candidate_losses)
            pareto_entries = [candidate_entries[i] for i in pareto_idx] if pareto_idx else []
            prepared.recorder.write_json("search/topk.json", top_entries)
            prepared.recorder.write_json("search/pareto.json", pareto_entries)
            report_lines = _build_report(
                cfg=prepared.cfg,
                grid_cfg=prepared.grid_cfg,
                baseline_eval=baseline_eval,
                baseline_metrics=baseline_metrics,
                candidate_entries=candidate_entries,
                pareto_entries=pareto_entries,
                sim_runs_failed=state.sim_runs_failed,
                param_value_errors=param_value_errors,
            )
            prepared.recorder.write_text("report.md", "\n".join(report_lines))

        recording_errors = finalize_run(
            recorder=prepared.recorder,
            manifest=prepared.manifest,
            best_source=state.best_source,
            best_metrics=state.best_metrics,
            history=prepared.history,
            stop_reason=stop_reason,
            best_score=state.best_score,
            best_iter=state.best_iter,
            sim_runs=state.sim_runs,
            sim_runs_ok=state.sim_runs_ok,
            sim_runs_failed=state.sim_runs_failed,
        )

        return RunResult(
            best_source=state.best_source,
            best_metrics=state.best_metrics,
            history=prepared.history,
            stop_reason=stop_reason,
            notes={"best_score": state.best_score, "all_pass": state.best_all_pass, "recording_errors": recording_errors},
        )

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:  # type: ignore[override]
        prepared = self._prepare(spec, source, ctx, cfg)
        baseline_out = self._baseline_phase(prepared, spec, ctx)
        if baseline_out.result is not None:
            return baseline_out.result

        assert baseline_out.baseline is not None
        assert baseline_out.eval0 is not None
        assert baseline_out.state is not None

        candidates_out = self._generate_candidates(prepared)

        if baseline_out.stop_reason is StopReason.reached_target:
            return self._write_report_and_finalize(
                prepared=prepared,
                state=baseline_out.state,
                stop_reason=baseline_out.stop_reason,
                baseline_eval=baseline_out.eval0,
                baseline_metrics=baseline_out.baseline.metrics,
                candidate_entries=[],
                candidate_losses=[],
                param_value_errors=candidates_out.param_value_errors,
            )

        candidate_entries, candidate_losses, stop_reason, state = self._evaluate_candidates_loop(
            prepared=prepared,
            spec=spec,
            ctx=ctx,
            state=baseline_out.state,
            candidates=candidates_out.candidates,
        )

        return self._write_report_and_finalize(
            prepared=prepared,
            state=state,
            stop_reason=stop_reason,
            baseline_eval=baseline_out.eval0,
            baseline_metrics=baseline_out.baseline.metrics,
            candidate_entries=candidate_entries,
            candidate_losses=candidate_losses,
            param_value_errors=candidates_out.param_value_errors,
        )


def _build_report(
    *,
    cfg: StrategyConfig,
    grid_cfg: Mapping[str, Any],
    baseline_eval: dict[str, Any],
    baseline_metrics: MetricsBundle,
    candidate_entries: list[dict[str, Any]],
    pareto_entries: list[dict[str, Any]],
    sim_runs_failed: int,
    param_value_errors: list[str],
) -> list[str]:
    lines: list[str] = []
    lines.append("# Grid Search Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- max_iterations: {cfg.budget.max_iterations}")
    lines.append(f"- mode: {grid_cfg.get('mode', 'coordinate')}")
    lines.append(f"- levels: {grid_cfg.get('levels', 10)}")
    lines.append(f"- span_mul: {grid_cfg.get('span_mul', 10.0)}")
    lines.append(f"- scale: {grid_cfg.get('scale', 'log')}")
    lines.append("")

    lines.append("## Baseline")
    lines.append(f"- score: {baseline_eval.get('score')}")
    lines.append(f"- all_pass: {baseline_eval.get('all_pass')}")
    for name, mv in baseline_metrics.values.items():
        lines.append(f"- metric {name}: {mv.value} {mv.unit or ''}".rstrip())
    if param_value_errors:
        lines.append("")
        lines.append("## Param Value Errors")
        for err in param_value_errors:
            lines.append(f"- {err}")

    lines.append("")
    lines.append("## Top-K Candidates")
    for entry in top_k(candidate_entries, int(grid_cfg.get("top_k", 5))):
        lines.append(
            f"- iter {entry.get('iteration')}: score={entry.get('score')} all_pass={entry.get('all_pass')} "
            f"candidate={entry.get('candidate')}"
        )

    lines.append("")
    lines.append("## Pareto Front")
    for entry in pareto_entries:
        lines.append(
            f"- iter {entry.get('iteration')}: score={entry.get('score')} losses={entry.get('losses')} "
            f"candidate={entry.get('candidate')}"
        )

    lines.append("")
    lines.append("## Failures")
    lines.append(f"- sim_runs_failed: {sim_runs_failed}")

    lines.append("")
    lines.append("## Files")
    lines.append("- search/candidates.json")
    lines.append("- search/topk.json")
    lines.append("- search/pareto.json")
    lines.append("- history/iterations.jsonl")
    lines.append("- best/best.sp")
    lines.append("- best/best_metrics.json")
    return lines
