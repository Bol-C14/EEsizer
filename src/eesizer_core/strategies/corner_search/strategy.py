from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...analysis.corners import aggregate_corner_results
from ...analysis.pareto import pareto_front, top_k
from ...contracts import (
    CircuitSource,
    CircuitSpec,
    MetricsBundle,
    Patch,
    PatchOp,
    RunResult,
    StrategyConfig,
)
from ...contracts.enums import PatchOpType, StopReason
from ...contracts.errors import ValidationError
from ...contracts.provenance import stable_hash_json, stable_hash_str
from ...contracts.strategy import Strategy
from ...domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ...domain.spice.patching import extract_param_values
from ...metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ...operators.guards import (
    BehaviorGuardOperator,
    GuardChainOperator,
    PatchGuardOperator,
    TopologyGuardOperator,
)
from ...operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ...operators.report_plots import ReportPlotsOperator
from ...runtime.recorder import RunRecorder
from ...runtime.recording_utils import (
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
from ...search.corners import build_corner_set
from ...search.ranges import RangeTrace
from ...search.samplers import coordinate_candidates, factorial_candidates, make_levels
from ...sim import DeckBuildOperator, NgspiceRunOperator
from ..attempt_pipeline import AttemptOperators, run_attempt
from ..patch_loop.evaluate import MeasureFn, evaluate_metrics, run_baseline
from ..patch_loop.planning import group_metric_names_by_kind, extract_sim_plan
from .config import parse_corner_search_config
from .measurement import (
    build_corner_patch,
    corner_result_dict,
    pick_corner,
    resolve_corner_overrides,
    stage_tag_for_corner,
)
from .report import build_corner_report
from .types import CornerSearchConfig


@dataclass
class _CornerPrepared:
    history: list[dict[str, Any]]
    recorder: RunRecorder | None
    manifest: Any
    cfg: StrategyConfig
    source: CircuitSource
    circuit_ir: Any
    signature: str
    param_space: Any
    guard_cfg: dict[str, Any]
    corner_cfg: CornerSearchConfig
    corner_cfg_payload: dict[str, Any]
    metric_groups: Mapping[Any, list[str]]
    sim_plan: Any | None
    max_iters: int
    max_sim_runs: int | None
    candidate_budget: int
    apply_opts: dict[str, Any]
    attempt_ops: AttemptOperators


@dataclass
class _CornerBaselineOutcome:
    result: RunResult | None
    baseline: Any | None
    eval0: dict[str, Any] | None


@dataclass
class _CornerParamSetup:
    search_param_ids: list[str]
    corner_param_ids: list[str]
    baseline_values: dict[str, float]
    corner_base_values: dict[str, float]
    param_value_errors: list[str]
    corner_set: dict[str, Any]
    corner_defs: list[dict[str, Any]]
    param_bounds: dict[str, Any]
    candidates: list[dict[str, float]]


@dataclass
class _CornerState:
    best_source: CircuitSource
    best_metrics: MetricsBundle
    best_score: float
    best_all_pass: bool
    best_iter: int
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int


@dataclass
class _CornerBaselineSummary:
    summary: dict[str, Any]
    corner_results: list[dict[str, Any]]
    baseline_corner_failed: bool
    state: _CornerState
    stop_reason: StopReason | None


@dataclass
class CornerSearchStrategy(Strategy):
    """Corner-aware grid/coordinate search strategy."""

    name: str = "corner_search"
    version: str = "0.1.1"

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

    def _prepare(
        self,
        spec: CircuitSpec,
        source: CircuitSource,
        ctx: Any,
        cfg: StrategyConfig,
    ) -> _CornerPrepared:
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
            manifest.files.setdefault("search/corner_set.json", "search/corner_set.json")
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

        corner_cfg = parse_corner_search_config(cfg.notes)
        corner_cfg_payload = {
            "mode": corner_cfg.mode,
            "levels": corner_cfg.levels,
            "span_mul": corner_cfg.span_mul,
            "scale": corner_cfg.scale,
            "top_k": corner_cfg.top_k,
            "stop_on_first_pass": corner_cfg.stop_on_first_pass,
            "baseline_retries": corner_cfg.baseline_retries,
            "corners": corner_cfg.corner_mode,
            "include_global_corners": corner_cfg.include_global_corners,
            "corner_override_mode": corner_cfg.override_mode,
            "require_baseline_corner_pass": corner_cfg.require_baseline_corner_pass,
            "clamp_corner_overrides": corner_cfg.clamp_corner_overrides,
            "allow_param_ids_override_frozen": corner_cfg.allow_param_ids_override_frozen,
        }

        max_iters = cfg.budget.max_iterations
        max_sim_runs = cfg.budget.max_sim_runs
        candidate_budget = max(0, max_iters - 1)

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)
        sim_plan = extract_sim_plan(spec.notes) or extract_sim_plan(cfg.notes)

        validation_opts = {
            "wl_ratio_min": guard_cfg.get("wl_ratio_min"),
            "max_mul_factor": guard_cfg.get("max_mul_factor", 10.0),
        }
        apply_opts = {
            "include_paths": cfg.notes.get("include_paths", True),
            "max_lines": cfg.notes.get("max_lines", 50000),
            "validation_opts": validation_opts,
        }

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

        return _CornerPrepared(
            history=history,
            recorder=recorder,
            manifest=manifest,
            cfg=cfg,
            source=source,
            circuit_ir=circuit_ir,
            signature=signature,
            param_space=param_space,
            guard_cfg=guard_cfg,
            corner_cfg=corner_cfg,
            corner_cfg_payload=corner_cfg_payload,
            metric_groups=metric_groups,
            sim_plan=sim_plan,
            max_iters=max_iters,
            max_sim_runs=max_sim_runs,
            candidate_budget=candidate_budget,
            apply_opts=apply_opts,
            attempt_ops=attempt_ops,
        )

    def _baseline_phase(
        self,
        prepared: _CornerPrepared,
        spec: CircuitSpec,
        ctx: Any,
    ) -> _CornerBaselineOutcome:
        baseline = run_baseline(
            source=prepared.source,
            spec=spec,
            metric_groups=prepared.metric_groups,
            ctx=ctx,
            guard_cfg=prepared.guard_cfg,
            max_retries=prepared.corner_cfg.baseline_retries,
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
                    "corners": [],
                    "pass_rate": 0.0,
                    "worst_score": float("inf"),
                    "robust_losses": [],
                    "worst_corner_id": None,
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
            return _CornerBaselineOutcome(
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
            )

        eval0 = evaluate_metrics(spec, baseline.metrics)
        return _CornerBaselineOutcome(result=None, baseline=baseline, eval0=eval0)

    def _resolve_param_ids(self, prepared: _CornerPrepared) -> tuple[list[str], list[str]]:
        search_param_ids = prepared.corner_cfg.search_param_ids
        if search_param_ids is None:
            search_param_ids = [p.param_id for p in prepared.param_space.params if not p.frozen]
        corner_param_ids = prepared.corner_cfg.corner_param_ids
        if corner_param_ids is None:
            corner_param_ids = list(search_param_ids)

        search_param_ids = [pid.lower() for pid in search_param_ids]
        corner_param_ids = [pid.lower() for pid in corner_param_ids]

        frozen_conflicts: list[str] = []
        for pid in search_param_ids:
            param_def = prepared.param_space.get(pid)
            if param_def is not None and param_def.frozen:
                frozen_conflicts.append(pid)
        for pid in corner_param_ids:
            param_def = prepared.param_space.get(pid)
            if param_def is not None and param_def.frozen:
                frozen_conflicts.append(pid)
        if frozen_conflicts and not prepared.corner_cfg.allow_param_ids_override_frozen:
            raise ValidationError(
                "corner_search param_ids include frozen params: "
                f"{', '.join(sorted(set(frozen_conflicts)))}; set allow_param_ids_override_frozen=true to override"
            )

        return search_param_ids, corner_param_ids

    def _build_corner_setup(self, prepared: _CornerPrepared) -> _CornerParamSetup:
        search_param_ids, corner_param_ids = self._resolve_param_ids(prepared)

        baseline_values, param_value_errors = extract_param_values(prepared.circuit_ir, param_ids=search_param_ids)
        corner_base_values, corner_value_errors = extract_param_values(prepared.circuit_ir, param_ids=corner_param_ids)
        if corner_value_errors:
            param_value_errors.extend(corner_value_errors)

        corner_set = build_corner_set(
            param_space=prepared.param_space,
            nominal_values=corner_base_values,
            span_mul=prepared.corner_cfg.span_mul,
            corner_param_ids=corner_param_ids,
            include_global_corners=prepared.corner_cfg.include_global_corners,
            override_mode=prepared.corner_cfg.override_mode,
            mode=prepared.corner_cfg.corner_mode,
        )
        corner_set["search_param_ids"] = list(search_param_ids)
        corner_set["corner_param_ids"] = list(corner_param_ids)
        if prepared.recorder is not None:
            prepared.recorder.write_json("search/corner_set.json", corner_set)

        per_param_levels: dict[str, list[float]] = {}
        ranges: list[RangeTrace] = []
        for param_id in search_param_ids:
            if param_id not in baseline_values:
                ranges.append(
                    RangeTrace(
                        param_id=param_id,
                        nominal=None,
                        lower=None,
                        upper=None,
                        scale=str(prepared.corner_cfg.scale or "log"),
                        levels=[],
                        source="unknown",
                        span_mul=prepared.corner_cfg.span_mul,
                        sanity={"warnings": []},
                        skipped=True,
                        skip_reason="missing_nominal",
                    )
                )
                continue
            param_def = prepared.param_space.get(param_id)
            if param_def is None:
                ranges.append(
                    RangeTrace(
                        param_id=param_id,
                        nominal=None,
                        lower=None,
                        upper=None,
                        scale=str(prepared.corner_cfg.scale or "log"),
                        levels=[],
                        source="unknown",
                        span_mul=prepared.corner_cfg.span_mul,
                        sanity={"warnings": []},
                        skipped=True,
                        skip_reason="param_not_found",
                    )
                )
                continue

            nominal = baseline_values.get(param_id)
            span_mul = float(prepared.corner_cfg.span_mul)
            lower = param_def.lower
            upper = param_def.upper
            source = ""
            warnings: list[str] = []
            clipped = False

            if lower is not None and upper is not None:
                source = "bounds"
            else:
                span = span_mul if span_mul > 0 else 1.0
                derived_lower = nominal / span if nominal != 0 else -span
                derived_upper = nominal * span if nominal != 0 else span
                if lower is None:
                    lower = derived_lower
                if upper is None:
                    upper = derived_upper
                source = "nominal*span_mul" if param_def.lower is None and param_def.upper is None else "bounds+span"

            if lower is not None and upper is not None and lower > upper:
                lower, upper = upper, lower
                clipped = True
                warnings.append("bounds_swapped")

            scale = str(prepared.corner_cfg.scale or "log").lower()
            if scale == "log" and (lower is None or upper is None or lower <= 0 or upper <= 0):
                scale = "linear"
                warnings.append("log_fallback_linear")

            levels = make_levels(
                nominal=nominal,
                lower=lower,
                upper=upper,
                levels=prepared.corner_cfg.levels,
                span_mul=span_mul,
                scale=scale,
            )
            per_param_levels[param_id] = levels
            ranges.append(
                RangeTrace(
                    param_id=param_id,
                    nominal=nominal,
                    lower=lower,
                    upper=upper,
                    scale=scale,
                    levels=list(levels),
                    source=source,
                    span_mul=span_mul,
                    sanity={
                        "positive_required": scale == "log",
                        "clipped": clipped,
                        "warnings": warnings,
                    },
                )
            )

        if prepared.corner_cfg.mode == "factorial":
            candidates = factorial_candidates(search_param_ids, per_param_levels, baseline_values)
        else:
            candidates = coordinate_candidates(search_param_ids, per_param_levels, baseline_values)

        total_generated = len(candidates)
        max_candidates = prepared.candidate_budget
        truncated = False
        if max_candidates is not None and total_generated > max_candidates:
            candidates = candidates[:max_candidates]
            truncated = True

        if prepared.recorder is not None:
            prepared.recorder.write_json("search/ranges.json", [r.to_dict() for r in ranges])
            prepared.recorder.write_json("search/candidates.json", candidates)
            prepared.recorder.write_json(
                "search/candidates_meta.json",
                {
                    "mode": prepared.corner_cfg.mode,
                    "include_nominal": False,
                    "total_generated": total_generated,
                    "truncated_to": len(candidates),
                    "max_candidates": max_candidates,
                    "truncate_policy": "lexicographic",
                    "seed": 0,
                    "truncated": truncated,
                },
            )

        corner_defs = corner_set.get("corners", []) or []
        param_bounds = corner_set.get("param_bounds", {})

        return _CornerParamSetup(
            search_param_ids=search_param_ids,
            corner_param_ids=corner_param_ids,
            baseline_values=baseline_values,
            corner_base_values=corner_base_values,
            param_value_errors=param_value_errors,
            corner_set=corner_set,
            corner_defs=corner_defs,
            param_bounds=param_bounds,
            candidates=candidates,
        )

    def _evaluate_baseline_corners(
        self,
        prepared: _CornerPrepared,
        setup: _CornerParamSetup,
        spec: CircuitSpec,
        ctx: Any,
        baseline: Any,
        eval0: dict[str, Any],
    ) -> _CornerBaselineSummary:
        state = _CornerState(
            best_source=prepared.source,
            best_metrics=baseline.metrics,
            best_score=eval0["score"],
            best_all_pass=eval0["all_pass"],
            best_iter=0,
            sim_runs=baseline.sim_runs,
            sim_runs_ok=baseline.sim_runs_ok,
            sim_runs_failed=baseline.sim_runs_failed,
        )

        corner_results: list[dict[str, Any]] = []
        corner_results.append(
            corner_result_dict(
                corner_id="nominal",
                overrides={},
                metrics=baseline.metrics,
                eval_result=eval0,
                stage_map=baseline.stage_map,
                warnings=baseline.warnings,
                errors=baseline.errors,
                guard_report=baseline.guard_report,
                recorder=prepared.recorder,
            )
        )

        stop_reason: StopReason | None = None
        baseline_corner_failed = False

        for corner in setup.corner_defs:
            corner_id = str(corner.get("corner_id", "corner"))
            if corner_id == "nominal":
                continue
            if prepared.max_sim_runs is not None and state.sim_runs >= prepared.max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                baseline_corner_failed = True
                corner_results.append(
                    corner_result_dict(
                        corner_id=corner_id,
                        overrides={},
                        metrics=None,
                        eval_result=None,
                        stage_map={},
                        warnings=[],
                        errors=["budget_exhausted"],
                        guard_report=None,
                        recorder=prepared.recorder,
                    )
                )
                break

            overrides = corner.get("overrides")
            if not isinstance(overrides, Mapping):
                overrides = {}
            applied, override_errors, override_warnings = resolve_corner_overrides(
                base_values=setup.corner_base_values,
                param_bounds=setup.param_bounds,
                overrides=overrides,
                clamp=prepared.corner_cfg.clamp_corner_overrides,
            )
            if override_errors:
                baseline_corner_failed = True
                corner_results.append(
                    corner_result_dict(
                        corner_id=corner_id,
                        overrides=applied,
                        metrics=None,
                        eval_result=None,
                        stage_map={},
                        warnings=override_warnings,
                        errors=override_errors,
                        guard_report=None,
                        recorder=prepared.recorder,
                    )
                )
                continue

            corner_patch = build_corner_patch(applied)
            corner_attempt = run_attempt(
                iter_idx=0,
                attempt=0,
                patch=corner_patch,
                cur_source=prepared.source,
                cur_signature=prepared.signature,
                circuit_ir=prepared.circuit_ir,
                param_space=prepared.param_space,
                spec=spec,
                guard_cfg=prepared.guard_cfg,
                apply_opts=prepared.apply_opts,
                metric_groups=prepared.metric_groups,
                ctx=ctx,
                recorder=prepared.recorder,
                manifest=prepared.manifest,
                measure_fn=self.measure_fn,
                ops=prepared.attempt_ops,
                stage_tag=stage_tag_for_corner(corner_id),
            )
            state.sim_runs += corner_attempt.sim_runs
            state.sim_runs_ok += corner_attempt.sim_runs_ok
            state.sim_runs_failed += corner_attempt.sim_runs_failed

            if not corner_attempt.success:
                baseline_corner_failed = True
            eval_result = (
                evaluate_metrics(spec, corner_attempt.metrics)
                if corner_attempt.metrics is not None and corner_attempt.success
                else None
            )
            corner_results.append(
                corner_result_dict(
                    corner_id=corner_id,
                    overrides=applied,
                    metrics=corner_attempt.metrics,
                    eval_result=eval_result,
                    stage_map=corner_attempt.stage_map,
                    warnings=override_warnings + corner_attempt.warnings,
                    errors=override_errors + corner_attempt.guard_failures,
                    guard_report=corner_attempt.guard_report,
                    recorder=prepared.recorder,
                )
            )

        summary0 = aggregate_corner_results(corner_results)
        worst_entry = pick_corner(corner_results, corner_id=summary0.get("worst_corner_id"), worst=True)
        worst_metrics = worst_entry.get("metrics") if worst_entry else {}
        worst_objectives = worst_entry.get("objectives") if worst_entry else []
        worst_stage_map = worst_entry.get("sim_stages") if worst_entry else {}
        worst_warnings = worst_entry.get("warnings") if worst_entry else []
        worst_errors = worst_entry.get("errors") if worst_entry else []

        state.best_score = summary0["worst_score"]
        state.best_all_pass = summary0["pass_rate"] == 1.0
        state.best_iter = 0

        if state.best_all_pass:
            stop_reason = StopReason.reached_target

        prepared.history.append(
            {
                "iteration": 0,
                "candidate": None,
                "patch": None,
                "signature_before": prepared.signature,
                "signature_after": prepared.signature,
                "metrics": worst_metrics,
                "score": summary0["worst_score"],
                "all_pass": state.best_all_pass,
                "improved": False,
                "objectives": worst_objectives,
                "sim_stages": worst_stage_map,
                "warnings": worst_warnings,
                "errors": worst_errors,
                "guard": guard_report_to_dict(baseline.guard_report) if baseline.guard_report else None,
                "attempts": baseline.attempts,
                "corners": corner_results,
                **summary0,
            }
        )
        record_history_entry(prepared.recorder, prepared.history[-1])

        return _CornerBaselineSummary(
            summary=summary0,
            corner_results=corner_results,
            baseline_corner_failed=baseline_corner_failed,
            state=state,
            stop_reason=stop_reason,
        )

    def _evaluate_candidates_loop(
        self,
        prepared: _CornerPrepared,
        setup: _CornerParamSetup,
        spec: CircuitSpec,
        ctx: Any,
        state: _CornerState,
    ) -> tuple[list[dict[str, Any]], list[list[float]], StopReason, _CornerState]:
        candidate_entries: list[dict[str, Any]] = []
        candidate_losses: list[list[float]] = []
        stop_reason: StopReason | None = None

        for idx, candidate in enumerate(setup.candidates, start=1):
            if prepared.max_sim_runs is not None and state.sim_runs >= prepared.max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            patch_ops = [
                PatchOp(param=pid, op=PatchOpType.set, value=val, why="corner_search")
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
                apply_opts=prepared.apply_opts,
                metric_groups=prepared.metric_groups,
                ctx=ctx,
                recorder=prepared.recorder,
                manifest=prepared.manifest,
                measure_fn=self.measure_fn,
                ops=prepared.attempt_ops,
                sim_plan=prepared.sim_plan,
                stage_tag=stage_tag_for_corner("nominal"),
            )
            state.sim_runs += attempt_result.sim_runs
            state.sim_runs_ok += attempt_result.sim_runs_ok
            state.sim_runs_failed += attempt_result.sim_runs_failed

            corner_results = []
            candidate_errors: list[str] = []
            nominal_eval = (
                evaluate_metrics(spec, attempt_result.metrics)
                if attempt_result.metrics is not None and attempt_result.success
                else None
            )
            corner_results.append(
                corner_result_dict(
                    corner_id="nominal",
                    overrides={},
                    metrics=attempt_result.metrics,
                    eval_result=nominal_eval,
                    stage_map=attempt_result.stage_map,
                    warnings=attempt_result.warnings,
                    errors=attempt_result.guard_failures,
                    guard_report=attempt_result.guard_report,
                    recorder=prepared.recorder,
                )
            )

            if not attempt_result.success or attempt_result.metrics is None:
                summary = aggregate_corner_results(corner_results)
                worst_entry = pick_corner(corner_results, corner_id=summary.get("worst_corner_id"), worst=True)
                worst_metrics = worst_entry.get("metrics") if worst_entry else {}
                worst_objectives = worst_entry.get("objectives") if worst_entry else []
                worst_stage_map = worst_entry.get("sim_stages") if worst_entry else {}
                worst_warnings = worst_entry.get("warnings") if worst_entry else []
                worst_errors = worst_entry.get("errors") if worst_entry else attempt_result.guard_failures

                prepared.history.append(
                    {
                        "iteration": idx,
                        "candidate": dict(candidate),
                        "patch": patch_to_dict(patch),
                        "signature_before": signature_before,
                        "signature_after": attempt_result.new_signature,
                        "metrics": worst_metrics,
                        "score": summary["worst_score"],
                        "all_pass": False,
                        "improved": False,
                        "objectives": worst_objectives,
                        "sim_stages": worst_stage_map,
                        "warnings": worst_warnings,
                        "errors": worst_errors,
                        "guard": guard_report_to_dict(attempt_result.guard_report)
                        if attempt_result.guard_report
                        else None,
                        "attempts": [
                            attempt_record(
                                0,
                                patch,
                                attempt_result.guard_report,
                                attempt_result.stage_map,
                                attempt_result.warnings,
                            )
                        ],
                        "corners": corner_results,
                        **summary,
                    }
                )
                record_history_entry(prepared.recorder, prepared.history[-1])
                continue

            candidate_base_values, candidate_value_errors = extract_param_values(
                attempt_result.new_circuit_ir,
                param_ids=setup.corner_param_ids,
            )
            if candidate_value_errors:
                candidate_errors.extend(candidate_value_errors)

            candidate_ok = True
            for corner in setup.corner_defs:
                corner_id = str(corner.get("corner_id", "corner"))
                if corner_id == "nominal":
                    continue
                if prepared.max_sim_runs is not None and state.sim_runs >= prepared.max_sim_runs:
                    stop_reason = StopReason.budget_exhausted
                    candidate_ok = False
                    corner_results.append(
                        corner_result_dict(
                            corner_id=corner_id,
                            overrides={},
                            metrics=None,
                            eval_result=None,
                            stage_map={},
                            warnings=[],
                            errors=["budget_exhausted"],
                            guard_report=None,
                            recorder=prepared.recorder,
                        )
                    )
                    break

                overrides = corner.get("overrides")
                if not isinstance(overrides, Mapping):
                    overrides = {}
                applied, override_errors, override_warnings = resolve_corner_overrides(
                    base_values=candidate_base_values,
                    param_bounds=setup.param_bounds,
                    overrides=overrides,
                    clamp=prepared.corner_cfg.clamp_corner_overrides,
                )
                if override_errors:
                    candidate_ok = False
                    candidate_errors.extend(override_errors)
                    corner_results.append(
                        corner_result_dict(
                            corner_id=corner_id,
                            overrides=applied,
                            metrics=None,
                            eval_result=None,
                            stage_map={},
                            warnings=override_warnings,
                            errors=override_errors,
                            guard_report=None,
                            recorder=prepared.recorder,
                        )
                    )
                    break

                corner_patch = build_corner_patch(applied)
                corner_attempt = run_attempt(
                    iter_idx=idx,
                    attempt=0,
                    patch=corner_patch,
                    cur_source=attempt_result.new_source,
                    cur_signature=attempt_result.new_signature,
                    circuit_ir=attempt_result.new_circuit_ir,
                    param_space=prepared.param_space,
                    spec=spec,
                    guard_cfg=prepared.guard_cfg,
                    apply_opts=prepared.apply_opts,
                    metric_groups=prepared.metric_groups,
                    ctx=ctx,
                    recorder=prepared.recorder,
                    manifest=prepared.manifest,
                    measure_fn=self.measure_fn,
                    ops=prepared.attempt_ops,
                    sim_plan=prepared.sim_plan,
                    stage_tag=stage_tag_for_corner(corner_id),
                )
                state.sim_runs += corner_attempt.sim_runs
                state.sim_runs_ok += corner_attempt.sim_runs_ok
                state.sim_runs_failed += corner_attempt.sim_runs_failed

                if not corner_attempt.success:
                    candidate_ok = False
                eval_result = (
                    evaluate_metrics(spec, corner_attempt.metrics)
                    if corner_attempt.metrics is not None and corner_attempt.success
                    else None
                )
                corner_results.append(
                    corner_result_dict(
                        corner_id=corner_id,
                        overrides=applied,
                        metrics=corner_attempt.metrics,
                        eval_result=eval_result,
                        stage_map=corner_attempt.stage_map,
                        warnings=override_warnings + corner_attempt.warnings,
                        errors=override_errors + corner_attempt.guard_failures,
                        guard_report=corner_attempt.guard_report,
                        recorder=prepared.recorder,
                    )
                )
                if not candidate_ok:
                    break

            summary = aggregate_corner_results(corner_results)
            worst_entry = pick_corner(corner_results, corner_id=summary.get("worst_corner_id"), worst=True)
            best_entry = pick_corner(corner_results, worst=False)
            nominal_entry = pick_corner(corner_results, corner_id="nominal", worst=False)
            worst_metrics = worst_entry.get("metrics") if worst_entry else {}
            worst_objectives = worst_entry.get("objectives") if worst_entry else []
            worst_stage_map = worst_entry.get("sim_stages") if worst_entry else {}
            worst_warnings = worst_entry.get("warnings") if worst_entry else []
            worst_errors = worst_entry.get("errors") if worst_entry else candidate_errors

            improved = candidate_ok and summary["worst_score"] < state.best_score - 1e-12
            if improved:
                state.best_score = summary["worst_score"]
                state.best_metrics = attempt_result.metrics if attempt_result.metrics is not None else state.best_metrics
                state.best_source = attempt_result.new_source
                state.best_all_pass = summary["pass_rate"] == 1.0
                state.best_iter = idx

            prepared.history.append(
                {
                    "iteration": idx,
                    "candidate": dict(candidate),
                    "patch": patch_to_dict(patch),
                    "signature_before": signature_before,
                    "signature_after": attempt_result.new_signature,
                    "metrics": worst_metrics,
                    "score": summary["worst_score"],
                    "all_pass": summary["pass_rate"] == 1.0,
                    "improved": improved,
                    "objectives": worst_objectives,
                    "sim_stages": worst_stage_map,
                    "warnings": worst_warnings,
                    "errors": worst_errors,
                    "guard": guard_report_to_dict(attempt_result.guard_report)
                    if attempt_result.guard_report
                    else None,
                    "attempts": [
                        attempt_record(
                            0,
                            patch,
                            attempt_result.guard_report,
                            attempt_result.stage_map,
                            attempt_result.warnings,
                        )
                    ],
                    "corners": corner_results,
                    **summary,
                }
            )
            record_history_entry(prepared.recorder, prepared.history[-1])

            if candidate_ok:
                candidate_entry = {
                    "iteration": idx,
                    "candidate": dict(candidate),
                    "score": summary["worst_score"],
                    "worst_score": summary["worst_score"],
                    "pass_rate": summary["pass_rate"],
                    "worst_corner_id": summary["worst_corner_id"],
                    "losses": list(summary["robust_losses"]),
                    "robust_losses": list(summary["robust_losses"]),
                    "nominal_metrics": nominal_entry.get("metrics") if nominal_entry else {},
                    "worst_metrics": worst_metrics,
                    "best_metrics": best_entry.get("metrics") if best_entry else {},
                }
                candidate_entries.append(candidate_entry)
                candidate_losses.append(list(summary["robust_losses"]))

            if stop_reason is StopReason.budget_exhausted:
                break
            if prepared.corner_cfg.stop_on_first_pass and summary["pass_rate"] == 1.0:
                stop_reason = StopReason.reached_target
                break

        if stop_reason is None:
            stop_reason = StopReason.max_iterations

        return candidate_entries, candidate_losses, stop_reason, state

    def _finalize(
        self,
        prepared: _CornerPrepared,
        state: _CornerState,
        stop_reason: StopReason,
    ) -> RunResult:
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
            notes={
                "best_score": state.best_score,
                "all_pass": state.best_all_pass,
                "recording_errors": recording_errors,
            },
        )

    def _write_report_and_finalize(
        self,
        prepared: _CornerPrepared,
        state: _CornerState,
        stop_reason: StopReason,
        baseline_eval: dict[str, Any],
        baseline_metrics: MetricsBundle,
        baseline_summary: dict[str, Any],
        candidate_entries: list[dict[str, Any]],
        candidate_losses: list[list[float]],
        param_value_errors: list[str],
    ) -> RunResult:
        if prepared.recorder is not None:
            top_entries = top_k(candidate_entries, prepared.corner_cfg.top_k)
            pareto_idx = pareto_front(candidate_losses)
            pareto_entries = [candidate_entries[i] for i in pareto_idx] if pareto_idx else []
            prepared.recorder.write_json("search/topk.json", top_entries)
            prepared.recorder.write_json("search/pareto.json", pareto_entries)
            report_lines = build_corner_report(
                cfg=prepared.cfg,
                corner_cfg=prepared.corner_cfg_payload,
                baseline_eval=baseline_eval,
                baseline_metrics=baseline_metrics,
                baseline_summary=baseline_summary,
                candidate_entries=candidate_entries,
                pareto_entries=pareto_entries,
                sim_runs_failed=state.sim_runs_failed,
                param_value_errors=param_value_errors,
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

        return self._finalize(prepared, state, stop_reason)

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:  # type: ignore[override]
        prepared = self._prepare(spec, source, ctx, cfg)
        baseline_out = self._baseline_phase(prepared, spec, ctx)
        if baseline_out.result is not None:
            return baseline_out.result

        assert baseline_out.baseline is not None
        assert baseline_out.eval0 is not None

        setup = self._build_corner_setup(prepared)
        baseline_summary = self._evaluate_baseline_corners(
            prepared=prepared,
            setup=setup,
            spec=spec,
            ctx=ctx,
            baseline=baseline_out.baseline,
            eval0=baseline_out.eval0,
        )
        state = baseline_summary.state

        if baseline_summary.stop_reason is StopReason.budget_exhausted:
            return self._finalize(prepared, state, baseline_summary.stop_reason)

        if prepared.corner_cfg.require_baseline_corner_pass and baseline_summary.baseline_corner_failed:
            return self._finalize(prepared, state, StopReason.guard_failed)

        if baseline_summary.stop_reason is StopReason.reached_target:
            return self._write_report_and_finalize(
                prepared=prepared,
                state=state,
                stop_reason=baseline_summary.stop_reason,
                baseline_eval=baseline_out.eval0,
                baseline_metrics=baseline_out.baseline.metrics,
                baseline_summary=baseline_summary.summary,
                candidate_entries=[],
                candidate_losses=[],
                param_value_errors=setup.param_value_errors,
            )

        candidate_entries, candidate_losses, stop_reason, state = self._evaluate_candidates_loop(
            prepared=prepared,
            setup=setup,
            spec=spec,
            ctx=ctx,
            state=state,
        )

        return self._write_report_and_finalize(
            prepared=prepared,
            state=state,
            stop_reason=stop_reason,
            baseline_eval=baseline_out.eval0,
            baseline_metrics=baseline_out.baseline.metrics,
            baseline_summary=baseline_summary.summary,
            candidate_entries=candidate_entries,
            candidate_losses=candidate_losses,
            param_value_errors=setup.param_value_errors,
        )
