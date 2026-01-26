from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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
from ..contracts.grid_search_config import GridSearchConfig
from ..contracts.errors import ValidationError
from ..contracts.enums import PatchOpType, StopReason
from ..contracts.strategy import Strategy
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ..domain.spice.patching import extract_param_values
from ..metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ..metrics.reporting import format_metric_line, metric_definition_lines
from ..operators.guards import (
    BehaviorGuardOperator,
    GuardChainOperator,
    PatchGuardOperator,
    TopologyGuardOperator,
)
from ..operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ..operators.report_plots import ReportPlotsOperator
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
from ..search.grid_config import parse_grid_search_config
from ..search.ranges import RangeTrace, generate_candidates, infer_ranges
from ..sim import DeckBuildOperator, NgspiceRunOperator
from .attempt_pipeline import AttemptOperators, run_attempt
from .patch_loop.evaluate import MeasureFn, evaluate_metrics, run_baseline
from .patch_loop.planning import group_metric_names_by_kind, extract_sim_plan


def _dedupe_ids(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        pid = str(item).strip().lower()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out


def _select_param_ids(param_space: Any, grid_cfg: GridSearchConfig) -> tuple[list[str], dict[str, Any]]:
    all_param_ids = [p.param_id for p in param_space.params]
    frozen_param_ids = [pid for pid in all_param_ids if param_space.get(pid) and param_space.get(pid).frozen]
    available_param_ids = [pid for pid in all_param_ids if pid not in frozen_param_ids]

    requested: list[str] = []
    source = "auto_truncate"
    warnings: list[str] = []

    if grid_cfg.param_select_policy == "recommended":
        if grid_cfg.recommended_knobs:
            requested = list(grid_cfg.recommended_knobs)
            source = "recommended_knobs"
        elif grid_cfg.param_ids:
            requested = list(grid_cfg.param_ids)
            source = "explicit_param_ids"
        else:
            warnings.append("no recommended_knobs/param_ids; using auto_truncate")
    elif grid_cfg.param_select_policy == "explicit":
        if grid_cfg.param_ids:
            requested = list(grid_cfg.param_ids)
            source = "explicit_param_ids"
        elif grid_cfg.recommended_knobs:
            requested = list(grid_cfg.recommended_knobs)
            source = "recommended_knobs_fallback"
            warnings.append("param_select_policy=explicit but param_ids missing; using recommended_knobs")
        else:
            warnings.append("param_select_policy=explicit but param_ids missing; using auto_truncate")
    else:
        source = "auto_truncate"

    if requested:
        selected = _dedupe_ids(requested)
    else:
        selected = sorted(available_param_ids)

    missing_param_ids = [pid for pid in selected if param_space.get(pid) is None]
    frozen_conflicts = [pid for pid in selected if pid in frozen_param_ids]

    if frozen_conflicts and not grid_cfg.allow_param_ids_override_frozen:
        raise ValidationError(
            "grid_search param_ids include frozen params: "
            f"{', '.join(sorted(set(frozen_conflicts)))}; set allow_param_ids_override_frozen=true to override"
        )

    if not grid_cfg.allow_param_ids_override_frozen:
        selected = [pid for pid in selected if pid not in frozen_param_ids]

    truncated = False
    truncated_param_ids: list[str] = []
    if grid_cfg.max_params and len(selected) > grid_cfg.max_params:
        truncated = True
        truncated_param_ids = selected[grid_cfg.max_params :]
        selected = selected[: grid_cfg.max_params]

    selection_info = {
        "policy": grid_cfg.param_select_policy,
        "source": source,
        "requested_param_ids": list(requested),
        "recommended_knobs": list(grid_cfg.recommended_knobs),
        "param_ids": list(selected),
        "max_params": grid_cfg.max_params,
        "truncated": truncated,
        "truncated_param_ids": truncated_param_ids,
        "missing_param_ids": missing_param_ids,
        "frozen_param_ids": frozen_param_ids,
        "allow_param_ids_override_frozen": grid_cfg.allow_param_ids_override_frozen,
        "warnings": warnings,
    }
    return selected, selection_info


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
    grid_cfg: GridSearchConfig
    metric_groups: Mapping[Any, list[str]]
    sim_plan: Any | None
    max_iters: int
    max_sim_runs: int | None
    candidate_budget: int


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
    ranges: list[RangeTrace]
    candidates_meta: dict[str, object]
    selection: dict[str, Any]
    nominal_values: dict[str, float]


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
    report_plots_op: Any = None

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
        if self.report_plots_op is None:
            self.report_plots_op = ReportPlotsOperator()

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
            manifest.files.setdefault("search/ranges.json", "search/ranges.json")
            manifest.files.setdefault("search/candidates_meta.json", "search/candidates_meta.json")
            manifest.files.setdefault("search/topk.json", "search/topk.json")
            manifest.files.setdefault("search/pareto.json", "search/pareto.json")
            manifest.files.setdefault("report.md", "report.md")
            if recorder is not None:
                recorder.write_input("source.sp", source.text)
                recorder.write_input("spec.json", spec_payload)
                recorder.write_input("param_space.json", param_payload)
                recorder.write_input("cfg.json", cfg_payload)
                recorder.write_input("signature.txt", signature)

        grid_cfg = parse_grid_search_config(cfg.notes, cfg.seed)

        max_iters = cfg.budget.max_iterations
        max_sim_runs = cfg.budget.max_sim_runs
        candidate_budget = max(0, max_iters - 1)

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)
        sim_plan = extract_sim_plan(spec.notes) or extract_sim_plan(cfg.notes)

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
            sim_plan=sim_plan,
            max_iters=max_iters,
            max_sim_runs=max_sim_runs,
            candidate_budget=candidate_budget,
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
            max_retries=prepared.grid_cfg.baseline_retries,
            max_sim_runs=prepared.max_sim_runs,
            recorder=prepared.recorder,
            manifest=prepared.manifest,
            measure_fn=self.measure_fn,
            deck_build_op=self.deck_build_op,
            sim_run_op=self.sim_run_op,
            metrics_op=self.metrics_op,
            behavior_guard_op=self.behavior_guard_op,
            guard_chain_op=self.guard_chain_op,
            sim_plan=prepared.sim_plan,
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
        param_ids, selection_info = _select_param_ids(prepared.param_space, prepared.grid_cfg)
        baseline_values, param_value_errors = extract_param_values(prepared.circuit_ir, param_ids=param_ids)
        ranges = infer_ranges(param_ids, prepared.param_space, baseline_values, prepared.grid_cfg)

        max_candidates = prepared.candidate_budget
        if prepared.grid_cfg.max_candidates is not None:
            max_candidates = min(prepared.grid_cfg.max_candidates, max_candidates)

        candidates, candidates_meta = generate_candidates(
            ranges,
            baseline_values,
            prepared.grid_cfg,
            max_candidates=max_candidates,
        )
        candidates_meta = dict(candidates_meta)
        candidates_meta.update(
            {
                "param_ids": list(selection_info.get("param_ids", [])),
                "selection_policy": selection_info.get("policy"),
                "selection_source": selection_info.get("source"),
            }
        )

        if prepared.recorder is not None:
            prepared.recorder.write_json("search/ranges.json", [r.to_dict() for r in ranges])
            prepared.recorder.write_json("search/candidates.json", candidates)
            prepared.recorder.write_json("search/candidates_meta.json", candidates_meta)

        return _GridCandidates(
            candidates=candidates,
            param_value_errors=param_value_errors,
            ranges=ranges,
            candidates_meta=candidates_meta,
            selection=selection_info,
            nominal_values=baseline_values,
        )

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
                sim_plan=prepared.sim_plan,
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

            if prepared.grid_cfg.stop_on_first_pass and eval_i["all_pass"]:
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
        ranges: list[RangeTrace],
        candidates_meta: dict[str, object],
        selection: dict[str, Any],
        nominal_values: Mapping[str, float],
    ) -> RunResult:
        if prepared.recorder is not None:
            top_entries = top_k(candidate_entries, prepared.grid_cfg.top_k)
            pareto_idx = pareto_front(candidate_losses)
            pareto_entries = [candidate_entries[i] for i in pareto_idx] if pareto_idx else []
            prepared.recorder.write_json("search/topk.json", top_entries)
            prepared.recorder.write_json("search/pareto.json", pareto_entries)
            failure_breakdown = _failure_breakdown(prepared.history)
            report_lines = _build_report(
                cfg=prepared.cfg,
                grid_cfg=prepared.grid_cfg,
                baseline_eval=baseline_eval,
                baseline_metrics=baseline_metrics,
                candidate_entries=candidate_entries,
                pareto_entries=pareto_entries,
                sim_runs_failed=state.sim_runs_failed,
                param_value_errors=param_value_errors,
                ranges=ranges,
                candidates_meta=candidates_meta,
                selection=selection,
                nominal_values=nominal_values,
                failure_breakdown=failure_breakdown,
            )
            prepared.recorder.write_text("report.md", "\n".join(report_lines))
            try:
                plot_result = self.report_plots_op.run(
                    {
                        "run_dir": prepared.recorder.run_dir,
                        "recorder": prepared.recorder,
                        "manifest": prepared.manifest,
                        "report_path": "report.md",
                    },
                    ctx=None,
                )
                record_operator_result(prepared.recorder, plot_result)
            except Exception:
                pass

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
                ranges=candidates_out.ranges,
                candidates_meta=candidates_out.candidates_meta,
                selection=candidates_out.selection,
                nominal_values=candidates_out.nominal_values,
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
            ranges=candidates_out.ranges,
            candidates_meta=candidates_out.candidates_meta,
            selection=candidates_out.selection,
            nominal_values=candidates_out.nominal_values,
        )


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.6g}"


def _format_levels(levels: Sequence[float]) -> str:
    return "[" + ", ".join(_format_float(val) for val in levels) + "]"


def _format_candidate_delta(candidate: Mapping[str, float], nominal_values: Mapping[str, float]) -> str:
    parts: list[str] = []
    for pid, value in candidate.items():
        nominal = nominal_values.get(pid)
        if nominal is None:
            parts.append(f"{pid}=unknown")
            continue
        if nominal == 0:
            parts.append(f"{pid}={_format_float(value)} (delta={_format_float(value)})")
            continue
        ratio = value / nominal
        parts.append(f"{pid}={_format_float(value)} ({ratio:.3g}x)")
    return ", ".join(parts)


def _failure_breakdown(history: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    breakdown = {"guard_fail": 0, "sim_fail": 0, "metric_missing": 0}
    for entry in history:
        if entry.get("iteration") in (None, 0):
            continue
        guard = entry.get("guard") or {}
        if not guard:
            continue
        if guard.get("ok", True):
            continue
        reasons: list[str] = []
        for check in guard.get("checks", []) or []:
            for reason in check.get("reasons", []) or []:
                reasons.append(str(reason))
        if any("metric '" in reason or "metric \"" in reason for reason in reasons):
            breakdown["metric_missing"] += 1
        elif any("measurement_failed" in reason or "SimulationError" in reason for reason in reasons):
            breakdown["sim_fail"] += 1
        else:
            breakdown["guard_fail"] += 1
    return breakdown


def _build_report(
    *,
    cfg: StrategyConfig,
    grid_cfg: GridSearchConfig,
    baseline_eval: dict[str, Any],
    baseline_metrics: MetricsBundle,
    candidate_entries: list[dict[str, Any]],
    pareto_entries: list[dict[str, Any]],
    sim_runs_failed: int,
    param_value_errors: list[str],
    ranges: Sequence[RangeTrace],
    candidates_meta: Mapping[str, object],
    selection: Mapping[str, Any],
    nominal_values: Mapping[str, float],
    failure_breakdown: Mapping[str, int],
) -> list[str]:
    lines: list[str] = []
    lines.append("# Grid Search Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- max_iterations: {cfg.budget.max_iterations}")
    lines.append(f"- mode: {grid_cfg.mode}")
    lines.append(f"- levels: {grid_cfg.levels}")
    lines.append(f"- span_mul: {grid_cfg.span_mul}")
    lines.append(f"- scale: {grid_cfg.scale}")
    lines.append(f"- include_nominal: {grid_cfg.include_nominal}")
    if grid_cfg.max_candidates is not None:
        lines.append(f"- max_candidates: {grid_cfg.max_candidates}")
    lines.append("")

    definition_lines = metric_definition_lines(baseline_metrics.values.keys())
    if definition_lines:
        lines.extend(definition_lines)
        lines.append("")

    lines.append("## Parameters Selected")
    lines.append(f"- policy: {selection.get('policy')}")
    lines.append(f"- source: {selection.get('source')}")
    lines.append(f"- param_ids: {selection.get('param_ids')}")
    if selection.get("truncated"):
        lines.append(f"- truncated: true; dropped={selection.get('truncated_param_ids')}")
    if selection.get("missing_param_ids"):
        lines.append(f"- missing_param_ids: {selection.get('missing_param_ids')}")
    if selection.get("warnings"):
        lines.append(f"- warnings: {selection.get('warnings')}")

    lines.append("")
    lines.append("## Ranges & Discretization")
    if not ranges:
        lines.append("- (none)")
    for trace in ranges:
        if trace.skipped:
            lines.append(f"- {trace.param_id}: skipped ({trace.skip_reason})")
            continue
        warnings = trace.sanity.get("warnings", [])
        warn_text = f", warnings={warnings}" if warnings else ""
        lines.append(
            f"- {trace.param_id}: nominal={_format_float(trace.nominal)} "
            f"lower={_format_float(trace.lower)} upper={_format_float(trace.upper)} "
            f"scale={trace.scale} levels={_format_levels(trace.levels)} source={trace.source}{warn_text}"
        )

    lines.append("")
    lines.append("## Candidate Generation")
    lines.append(f"- mode: {candidates_meta.get('mode')}")
    lines.append(f"- include_nominal: {candidates_meta.get('include_nominal')}")
    lines.append(f"- total_generated: {candidates_meta.get('total_generated')}")
    lines.append(f"- truncated_to: {candidates_meta.get('truncated_to')}")
    lines.append(f"- max_candidates: {candidates_meta.get('max_candidates')}")
    lines.append(f"- truncate_policy: {candidates_meta.get('truncate_policy')}")
    lines.append(f"- seed: {candidates_meta.get('seed')}")

    lines.append("")
    lines.append("## Baseline")
    lines.append(f"- score: {baseline_eval.get('score')}")
    lines.append(f"- all_pass: {baseline_eval.get('all_pass')}")
    for name, mv in baseline_metrics.values.items():
        lines.append(format_metric_line(name, mv))
    if param_value_errors:
        lines.append("")
        lines.append("## Param Value Errors")
        for err in param_value_errors:
            lines.append(f"- {err}")

    lines.append("")
    lines.append("## Top-K Candidates")
    top_entries = top_k(candidate_entries, grid_cfg.top_k)
    if not top_entries:
        lines.append("- (none)")
    for entry in top_entries:
        delta = _format_candidate_delta(entry.get("candidate", {}), nominal_values)
        lines.append(
            f"- iter {entry.get('iteration')}: score={entry.get('score')} all_pass={entry.get('all_pass')} "
            f"candidate={entry.get('candidate')} delta={delta}"
        )

    lines.append("")
    lines.append("## Pareto Front")
    if not pareto_entries:
        lines.append("- (none)")
    for entry in pareto_entries:
        delta = _format_candidate_delta(entry.get("candidate", {}), nominal_values)
        lines.append(
            f"- iter {entry.get('iteration')}: score={entry.get('score')} losses={entry.get('losses')} "
            f"candidate={entry.get('candidate')} delta={delta}"
        )

    lines.append("")
    lines.append("## Failures")
    lines.append(f"- sim_runs_failed: {sim_runs_failed}")
    lines.append(f"- guard_failures: {failure_breakdown.get('guard_fail', 0)}")
    lines.append(f"- sim_failures: {failure_breakdown.get('sim_fail', 0)}")
    lines.append(f"- metric_missing: {failure_breakdown.get('metric_missing', 0)}")

    lines.append("")
    lines.append("## Files")
    lines.append("- search/candidates.json")
    lines.append("- search/ranges.json")
    lines.append("- search/candidates_meta.json")
    lines.append("- search/topk.json")
    lines.append("- search/pareto.json")
    lines.append("- history/iterations.jsonl")
    lines.append("- best/best.sp")
    lines.append("- best/best_metrics.json")
    return lines
