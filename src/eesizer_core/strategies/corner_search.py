from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..analysis.corners import aggregate_corner_results
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
from ..contracts.enums import PatchOpType, SimKind, StopReason
from ..contracts.errors import MetricError, SimulationError, ValidationError
from ..contracts.guards import GuardCheck, GuardReport
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..contracts.strategy import Strategy
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
    guard_failures,
    guard_report_to_dict,
    param_space_to_dict,
    patch_to_dict,
    record_history_entry,
    record_operator_result,
    spec_to_dict,
    strategy_cfg_to_dict,
)
from ..search.corners import build_corner_set
from ..search.samplers import coordinate_candidates, factorial_candidates, make_levels
from ..sim import DeckBuildOperator, NgspiceRunOperator
from .patch_loop.evaluate import MeasureFn, evaluate_metrics, run_baseline
from .patch_loop.planning import group_metric_names_by_kind, merge_metrics, sim_plan_for_kind


_STAGE_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


def _corner_cfg(notes: Mapping[str, Any]) -> dict[str, Any]:
    raw = notes.get("corner_search")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _stage_suffix(corner_id: str) -> str:
    text = str(corner_id or "").strip().lower()
    if not text:
        return "corner"
    return _STAGE_SAFE.sub("_", text)


def _metrics_values(metrics: MetricsBundle) -> dict[str, Any]:
    return {name: mv.value for name, mv in metrics.values.items()}


def _relativize_stage_map(stage_map: Mapping[str, str], recorder: RunRecorder | None) -> dict[str, str]:
    if recorder is None:
        return {str(k): str(v) for k, v in stage_map.items()}
    return {str(k): recorder.relpath(v) for k, v in stage_map.items()}


@dataclass
class CornerMeasurement:
    metrics: MetricsBundle
    stage_map: dict[str, str]
    warnings: list[str]
    sim_runs: int


def _measure_corner_metrics(
    *,
    source: CircuitSource,
    corner_id: str,
    metric_groups: Mapping[SimKind, list[str]],
    ctx: Any,
    iter_idx: int,
    attempt_idx: int,
    recorder: RunRecorder | None,
    manifest: Any,
    measure_fn: MeasureFn | None,
    deck_build_op: Any,
    sim_run_op: Any,
    metrics_op: Any,
) -> CornerMeasurement:
    if measure_fn is not None:
        return CornerMeasurement(metrics=measure_fn(source, iter_idx), stage_map={}, warnings=[], sim_runs=0)

    bundles: list[MetricsBundle] = []
    stage_map: dict[str, str] = {}
    warnings: list[str] = []
    sim_runs = 0
    corner_suffix = _stage_suffix(corner_id)

    for kind, names in metric_groups.items():
        plan = sim_plan_for_kind(kind)
        deck_res = deck_build_op.run({"circuit_source": source, "sim_plan": plan, "sim_kind": kind}, ctx=None)
        record_operator_result(recorder, deck_res)
        deck = deck_res.outputs["deck"]
        stage_name = f"{kind.value}_{corner_suffix}_i{iter_idx:03d}_a{attempt_idx:02d}"
        run_res = sim_run_op.run({"deck": deck, "stage": stage_name}, ctx)
        record_operator_result(recorder, run_res)
        if manifest is not None:
            version = run_res.provenance.notes.get("ngspice_version")
            path = run_res.provenance.notes.get("ngspice_path")
            if version:
                manifest.environment.setdefault("ngspice_version", version)
            if path:
                manifest.environment.setdefault("ngspice_path", path)
        raw = run_res.outputs["raw_data"]
        metrics_res = metrics_op.run({"raw_data": raw, "metric_names": names}, ctx=None)
        record_operator_result(recorder, metrics_res)
        bundles.append(metrics_res.outputs["metrics"])
        stage_map[kind.value] = str(raw.run_dir)
        warnings.extend(deck_res.warnings)
        warnings.extend(run_res.warnings)
        sim_runs += 1

    return CornerMeasurement(metrics=merge_metrics(bundles), stage_map=stage_map, warnings=warnings, sim_runs=sim_runs)


def _corner_summary_defaults() -> dict[str, Any]:
    return {
        "pass_rate": 0.0,
        "worst_score": float("inf"),
        "robust_losses": [],
        "worst_corner_id": None,
    }


def _corner_patch(overrides: Mapping[str, float]) -> Patch:
    ops = [
        PatchOp(param=param_id, op=PatchOpType.set, value=value, why="corner_override")
        for param_id, value in overrides.items()
    ]
    return Patch(ops=tuple(ops))


def _corner_result(
    *,
    corner_id: str,
    overrides: Mapping[str, float],
    metrics: MetricsBundle | None,
    eval_result: dict[str, Any] | None,
    stage_map: Mapping[str, str],
    warnings: list[str],
    errors: list[str],
    guard_report: GuardReport | None,
    recorder: RunRecorder | None,
) -> dict[str, Any]:
    metrics_payload = _metrics_values(metrics) if metrics is not None else {}
    losses = objective_losses(eval_result or {})
    return {
        "corner_id": corner_id,
        "overrides": dict(overrides),
        "metrics": metrics_payload,
        "score": eval_result.get("score", float("inf")) if eval_result is not None else float("inf"),
        "all_pass": bool(eval_result.get("all_pass")) if eval_result is not None else False,
        "losses": list(losses),
        "objectives": eval_result.get("per_objective", []) if eval_result is not None else [],
        "sim_stages": _relativize_stage_map(stage_map, recorder),
        "warnings": list(warnings),
        "errors": list(errors),
        "guard": guard_report_to_dict(guard_report) if guard_report else None,
    }


def _pick_corner(corners: list[dict[str, Any]], *, corner_id: str | None = None, worst: bool = False) -> dict[str, Any] | None:
    if not corners:
        return None
    if corner_id is not None:
        for entry in corners:
            if entry.get("corner_id") == corner_id:
                return entry
    def _score(entry: Mapping[str, Any]) -> float:
        try:
            return float(entry.get("score", float("inf")))
        except (TypeError, ValueError):
            return float("inf")
    return max(corners, key=_score) if worst else min(corners, key=_score)


@dataclass
class CornerSearchStrategy(Strategy):
    """Corner-aware grid/coordinate search strategy."""

    name: str = "corner_search"
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

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:  # type: ignore[override]
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
            manifest.files.setdefault("search/topk.json", "search/topk.json")
            manifest.files.setdefault("search/pareto.json", "search/pareto.json")
            manifest.files.setdefault("report.md", "report.md")
            if recorder is not None:
                recorder.write_input("source.sp", source.text)
                recorder.write_input("spec.json", spec_payload)
                recorder.write_input("param_space.json", param_payload)
                recorder.write_input("cfg.json", cfg_payload)
                recorder.write_input("signature.txt", signature)

        corner_cfg = _corner_cfg(cfg.notes)
        mode = str(corner_cfg.get("mode", "coordinate")).lower()
        levels = int(corner_cfg.get("levels", 10))
        span_mul = float(corner_cfg.get("span_mul", 10.0))
        scale = str(corner_cfg.get("scale", "log")).lower()
        top_k_count = int(corner_cfg.get("top_k", 5))
        stop_on_first_pass = bool(corner_cfg.get("stop_on_first_pass", False))
        baseline_retries = int(corner_cfg.get("baseline_retries", 0))
        corner_mode = str(corner_cfg.get("corners", "oat")).lower()

        max_iters = cfg.budget.max_iterations
        max_sim_runs = cfg.budget.max_sim_runs
        candidate_budget = max(0, max_iters - 1)

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)

        baseline = run_baseline(
            source=source,
            spec=spec,
            metric_groups=metric_groups,
            ctx=ctx,
            guard_cfg=guard_cfg,
            max_retries=baseline_retries,
            max_sim_runs=max_sim_runs,
            recorder=recorder,
            manifest=manifest,
            measure_fn=self.measure_fn,
            deck_build_op=self.deck_build_op,
            sim_run_op=self.sim_run_op,
            metrics_op=self.metrics_op,
            behavior_guard_op=self.behavior_guard_op,
            guard_chain_op=self.guard_chain_op,
        )

        if not baseline.success:
            summary = _corner_summary_defaults()
            history.append(
                {
                    "iteration": 0,
                    "candidate": None,
                    "patch": None,
                    "signature_before": signature,
                    "signature_after": signature,
                    "metrics": {},
                    "score": summary["worst_score"],
                    "all_pass": False,
                    "improved": False,
                    "objectives": [],
                    "sim_stages": baseline.stage_map,
                    "warnings": baseline.warnings,
                    "errors": baseline.errors,
                    "guard": guard_report_to_dict(baseline.guard_report) if baseline.guard_report else None,
                    "attempts": baseline.attempts,
                    "corners": [],
                    **summary,
                }
            )
            record_history_entry(recorder, history[-1])
            recording_errors = finalize_run(
                recorder=recorder,
                manifest=manifest,
                best_source=source,
                best_metrics=MetricsBundle(),
                history=history,
                stop_reason=baseline.stop_reason,
                best_score=summary["worst_score"],
                best_iter=None,
                sim_runs=baseline.sim_runs,
                sim_runs_ok=baseline.sim_runs_ok,
                sim_runs_failed=baseline.sim_runs_failed,
            )
            return RunResult(
                best_source=source,
                best_metrics=MetricsBundle(),
                history=history,
                stop_reason=baseline.stop_reason,
                notes={
                    "best_score": summary["worst_score"],
                    "all_pass": False,
                    "recording_errors": recording_errors,
                },
            )

        eval0 = evaluate_metrics(spec, baseline.metrics)

        param_ids = [p.param_id for p in param_space.params]
        cfg_param_ids = corner_cfg.get("param_ids")
        if isinstance(cfg_param_ids, (list, tuple)):
            param_ids = [str(pid).lower() for pid in cfg_param_ids]

        baseline_values, param_value_errors = extract_param_values(circuit_ir, param_ids=param_ids)
        corner_set = build_corner_set(
            param_space=param_space,
            nominal_values=baseline_values,
            span_mul=span_mul,
            param_ids=param_ids,
            mode=corner_mode,
        )
        if recorder is not None:
            recorder.write_json("search/corner_set.json", corner_set)

        per_param_levels: dict[str, list[float]] = {}
        for param_id in param_ids:
            if param_id not in baseline_values:
                continue
            param_def = param_space.get(param_id)
            if param_def is None:
                continue
            per_param_levels[param_id] = make_levels(
                nominal=baseline_values[param_id],
                lower=param_def.lower,
                upper=param_def.upper,
                levels=levels,
                span_mul=span_mul,
                scale=scale,
            )

        if mode == "factorial":
            candidates = factorial_candidates(param_ids, per_param_levels, baseline_values)
        else:
            candidates = coordinate_candidates(param_ids, per_param_levels, baseline_values)

        if candidate_budget and len(candidates) > candidate_budget:
            candidates = candidates[:candidate_budget]

        if recorder is not None:
            recorder.write_json("search/candidates.json", candidates)

        validation_opts = {
            "wl_ratio_min": guard_cfg.get("wl_ratio_min"),
            "max_mul_factor": guard_cfg.get("max_mul_factor", 10.0),
        }
        apply_opts = {
            "include_paths": cfg.notes.get("include_paths", True),
            "max_lines": cfg.notes.get("max_lines", 50000),
            "validation_opts": validation_opts,
        }

        sim_runs = baseline.sim_runs
        sim_runs_ok = baseline.sim_runs_ok
        sim_runs_failed = baseline.sim_runs_failed

        corner_results: list[dict[str, Any]] = []
        corner_results.append(
            _corner_result(
                corner_id="nominal",
                overrides={},
                metrics=baseline.metrics,
                eval_result=eval0,
                stage_map=baseline.stage_map,
                warnings=baseline.warnings,
                errors=baseline.errors,
                guard_report=baseline.guard_report,
                recorder=recorder,
            )
        )
        baseline_corner_ok = True
        stop_reason: StopReason | None = None

        for corner in corner_set.get("corners", []) or []:
            corner_id = str(corner.get("corner_id", "corner"))
            if corner_id == "nominal":
                continue
            overrides = corner.get("overrides")
            if not isinstance(overrides, Mapping):
                overrides = {}

            if max_sim_runs is not None and sim_runs >= max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                baseline_corner_ok = False
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=None,
                        eval_result=None,
                        stage_map={},
                        warnings=[],
                        errors=["budget_exhausted"],
                        guard_report=None,
                        recorder=recorder,
                    )
                )
                break

            corner_patch = _corner_patch(overrides)
            checks: list[GuardCheck] = []
            guard_report: GuardReport | None = None
            errors: list[str] = []
            stage_map: dict[str, str] = {}
            warnings: list[str] = []
            metrics_i: MetricsBundle | None = None
            eval_i: dict[str, Any] | None = None

            pre_check_res = self.patch_guard_op.run(
                {
                    "circuit_ir": circuit_ir,
                    "param_space": param_space,
                    "patch": corner_patch,
                    "spec": spec,
                    "guard_cfg": guard_cfg,
                },
                ctx=None,
            )
            record_operator_result(recorder, pre_check_res)
            pre_check = pre_check_res.outputs["check"]
            checks.append(pre_check)
            if not pre_check.ok:
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                errors = guard_failures(guard_report)
                baseline_corner_ok = False
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=None,
                        eval_result=None,
                        stage_map=stage_map,
                        warnings=warnings,
                        errors=errors,
                        guard_report=guard_report,
                        recorder=recorder,
                    )
                )
                break

            try:
                apply_res = self.patch_apply_op.run(
                    {"source": source, "param_space": param_space, "patch": corner_patch, **apply_opts},
                    ctx=None,
                )
                record_operator_result(recorder, apply_res)
                apply_outputs = apply_res.outputs
                corner_source = apply_outputs["source"]
                corner_signature = apply_outputs["topology_signature"]
            except ValidationError as exc:
                check_name = "topology_guard" if "Topology changed" in str(exc) else "patch_guard"
                checks.append(
                    GuardCheck(
                        name=check_name,
                        ok=False,
                        severity="hard",
                        reasons=(str(exc),),
                    )
                )
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                errors = guard_failures(guard_report)
                baseline_corner_ok = False
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=None,
                        eval_result=None,
                        stage_map=stage_map,
                        warnings=warnings,
                        errors=errors,
                        guard_report=guard_report,
                        recorder=recorder,
                    )
                )
                break

            topo_check_res = self.topology_guard_op.run(
                {"signature_before": signature, "signature_after": corner_signature},
                ctx=None,
            )
            record_operator_result(recorder, topo_check_res)
            topo_check = topo_check_res.outputs["check"]
            checks.append(topo_check)
            if not topo_check.ok:
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                errors = guard_failures(guard_report)
                baseline_corner_ok = False
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=None,
                        eval_result=None,
                        stage_map=stage_map,
                        warnings=warnings,
                        errors=errors,
                        guard_report=guard_report,
                        recorder=recorder,
                    )
                )
                break

            try:
                measurement = _measure_corner_metrics(
                    source=corner_source,
                    corner_id=corner_id,
                    metric_groups=metric_groups,
                    ctx=ctx,
                    iter_idx=0,
                    attempt_idx=0,
                    recorder=recorder,
                    manifest=manifest,
                    measure_fn=self.measure_fn,
                    deck_build_op=self.deck_build_op,
                    sim_run_op=self.sim_run_op,
                    metrics_op=self.metrics_op,
                )
                metrics_i = measurement.metrics
                stage_map = measurement.stage_map
                warnings = measurement.warnings
                sim_runs += measurement.sim_runs
                sim_runs_ok += measurement.sim_runs
            except (SimulationError, MetricError, ValidationError) as exc:
                sim_runs += 1
                sim_runs_failed += 1
                checks.append(
                    GuardCheck(
                        name="behavior_guard",
                        ok=False,
                        severity="hard",
                        reasons=(str(exc),),
                        data={"error_type": type(exc).__name__},
                    )
                )
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                errors = guard_failures(guard_report)
                baseline_corner_ok = False
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=None,
                        eval_result=None,
                        stage_map=stage_map,
                        warnings=warnings,
                        errors=errors,
                        guard_report=guard_report,
                        recorder=recorder,
                    )
                )
                break

            behavior_check_res = self.behavior_guard_op.run(
                {"metrics": metrics_i, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg},
                ctx=None,
            )
            record_operator_result(recorder, behavior_check_res)
            behavior_check = behavior_check_res.outputs["check"]
            checks.append(behavior_check)
            guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
            record_operator_result(recorder, guard_chain_res)
            guard_report = guard_chain_res.outputs["report"]
            errors = guard_failures(guard_report)

            if not guard_report.ok:
                baseline_corner_ok = False
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=metrics_i,
                        eval_result=None,
                        stage_map=stage_map,
                        warnings=warnings,
                        errors=errors,
                        guard_report=guard_report,
                        recorder=recorder,
                    )
                )
                break

            eval_i = evaluate_metrics(spec, metrics_i)
            corner_results.append(
                _corner_result(
                    corner_id=corner_id,
                    overrides=overrides,
                    metrics=metrics_i,
                    eval_result=eval_i,
                    stage_map=stage_map,
                    warnings=warnings,
                    errors=errors,
                    guard_report=guard_report,
                    recorder=recorder,
                )
            )

        if not baseline_corner_ok:
            summary = _corner_summary_defaults()
            history.append(
                {
                    "iteration": 0,
                    "candidate": None,
                    "patch": None,
                    "signature_before": signature,
                    "signature_after": signature,
                    "metrics": {},
                    "score": summary["worst_score"],
                    "all_pass": False,
                    "improved": False,
                    "objectives": [],
                    "sim_stages": baseline.stage_map,
                    "warnings": baseline.warnings,
                    "errors": ["corner_eval_failed"],
                    "guard": guard_report_to_dict(baseline.guard_report) if baseline.guard_report else None,
                    "attempts": baseline.attempts,
                    "corners": corner_results,
                    **summary,
                }
            )
            record_history_entry(recorder, history[-1])
            recording_errors = finalize_run(
                recorder=recorder,
                manifest=manifest,
                best_source=source,
                best_metrics=baseline.metrics,
                history=history,
                stop_reason=stop_reason or StopReason.guard_failed,
                best_score=summary["worst_score"],
                best_iter=None,
                sim_runs=sim_runs,
                sim_runs_ok=sim_runs_ok,
                sim_runs_failed=sim_runs_failed,
            )
            return RunResult(
                best_source=source,
                best_metrics=baseline.metrics,
                history=history,
                stop_reason=stop_reason or StopReason.guard_failed,
                notes={
                    "best_score": summary["worst_score"],
                    "all_pass": False,
                    "recording_errors": recording_errors,
                },
            )

        summary0 = aggregate_corner_results(corner_results)
        worst_entry = _pick_corner(corner_results, corner_id=summary0.get("worst_corner_id"), worst=True)
        best_entry = _pick_corner(corner_results, worst=False)
        nominal_entry = _pick_corner(corner_results, corner_id="nominal", worst=False)
        worst_metrics = worst_entry.get("metrics") if worst_entry else {}
        worst_objectives = worst_entry.get("objectives") if worst_entry else []
        worst_stage_map = worst_entry.get("sim_stages") if worst_entry else {}
        worst_warnings = worst_entry.get("warnings") if worst_entry else []
        worst_errors = worst_entry.get("errors") if worst_entry else []

        best_source = source
        best_metrics = baseline.metrics
        best_score = summary0["worst_score"]
        best_all_pass = summary0["pass_rate"] == 1.0
        best_iter = 0
        stop_reason = StopReason.reached_target if best_all_pass else None

        history.append(
            {
                "iteration": 0,
                "candidate": None,
                "patch": None,
                "signature_before": signature,
                "signature_after": signature,
                "metrics": worst_metrics,
                "score": summary0["worst_score"],
                "all_pass": best_all_pass,
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
        record_history_entry(recorder, history[-1])

        if stop_reason is StopReason.reached_target:
            recording_errors = finalize_run(
                recorder=recorder,
                manifest=manifest,
                best_source=best_source,
                best_metrics=best_metrics,
                history=history,
                stop_reason=stop_reason,
                best_score=best_score,
                best_iter=best_iter,
                sim_runs=sim_runs,
                sim_runs_ok=sim_runs_ok,
                sim_runs_failed=sim_runs_failed,
            )
            return RunResult(
                best_source=best_source,
                best_metrics=best_metrics,
                history=history,
                stop_reason=stop_reason,
                notes={"best_score": best_score, "all_pass": best_all_pass, "recording_errors": recording_errors},
            )

        candidate_entries: list[dict[str, Any]] = []
        candidate_losses: list[list[float]] = []

        for idx, candidate in enumerate(candidates, start=1):
            if max_sim_runs is not None and sim_runs >= max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            patch_ops = [
                PatchOp(param=pid, op=PatchOpType.set, value=val, why="corner_search")
                for pid, val in candidate.items()
            ]
            patch = Patch(ops=tuple(patch_ops))
            signature_before = signature
            new_source = source
            new_signature = signature
            new_circuit_ir = circuit_ir
            attempts: list[dict[str, Any]] = []
            errors: list[str] = []
            guard_report: GuardReport | None = None
            checks: list[GuardCheck] = []

            pre_check_res = self.patch_guard_op.run(
                {
                    "circuit_ir": circuit_ir,
                    "param_space": param_space,
                    "patch": patch,
                    "spec": spec,
                    "guard_cfg": guard_cfg,
                },
                ctx=None,
            )
            record_operator_result(recorder, pre_check_res)
            pre_check = pre_check_res.outputs["check"]
            checks.append(pre_check)
            if not pre_check.ok:
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                errors = guard_failures(guard_report)
                attempts.append(attempt_record(0, patch, guard_report))
                history.append(
                    {
                        "iteration": idx,
                        "candidate": dict(candidate),
                        "patch": patch_to_dict(patch),
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {},
                        "score": float("inf"),
                        "all_pass": False,
                        "improved": False,
                        "objectives": [],
                        "sim_stages": {},
                        "warnings": [],
                        "errors": errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                        "corners": [],
                        **_corner_summary_defaults(),
                    }
                )
                record_history_entry(recorder, history[-1])
                continue

            try:
                apply_res = self.patch_apply_op.run(
                    {"source": source, "param_space": param_space, "patch": patch, **apply_opts},
                    ctx=None,
                )
                record_operator_result(recorder, apply_res)
                apply_outputs = apply_res.outputs
                new_source = apply_outputs["source"]
                new_signature = apply_outputs["topology_signature"]
                new_circuit_ir = apply_outputs["circuit_ir"]
            except ValidationError as exc:
                check_name = "topology_guard" if "Topology changed" in str(exc) else "patch_guard"
                checks.append(
                    GuardCheck(
                        name=check_name,
                        ok=False,
                        severity="hard",
                        reasons=(str(exc),),
                    )
                )
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                errors = guard_failures(guard_report)
                attempts.append(attempt_record(0, patch, guard_report))
                history.append(
                    {
                        "iteration": idx,
                        "candidate": dict(candidate),
                        "patch": patch_to_dict(patch),
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {},
                        "score": float("inf"),
                        "all_pass": False,
                        "improved": False,
                        "objectives": [],
                        "sim_stages": {},
                        "warnings": [],
                        "errors": errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                        "corners": [],
                        **_corner_summary_defaults(),
                    }
                )
                record_history_entry(recorder, history[-1])
                continue

            topo_check_res = self.topology_guard_op.run(
                {"signature_before": signature_before, "signature_after": new_signature},
                ctx=None,
            )
            record_operator_result(recorder, topo_check_res)
            topo_check = topo_check_res.outputs["check"]
            checks.append(topo_check)
            guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
            record_operator_result(recorder, guard_chain_res)
            guard_report = guard_chain_res.outputs["report"]
            attempts.append(attempt_record(0, patch, guard_report))
            errors = guard_failures(guard_report)

            if not guard_report.ok:
                history.append(
                    {
                        "iteration": idx,
                        "candidate": dict(candidate),
                        "patch": patch_to_dict(patch),
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {},
                        "score": float("inf"),
                        "all_pass": False,
                        "improved": False,
                        "objectives": [],
                        "sim_stages": {},
                        "warnings": [],
                        "errors": errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                        "corners": [],
                        **_corner_summary_defaults(),
                    }
                )
                record_history_entry(recorder, history[-1])
                continue

            corner_results = []
            corner_metrics_bundle: dict[str, MetricsBundle] = {}
            candidate_ok = True
            candidate_errors: list[str] = []

            for corner in corner_set.get("corners", []) or []:
                corner_id = str(corner.get("corner_id", "corner"))
                overrides = corner.get("overrides")
                if not isinstance(overrides, Mapping):
                    overrides = {}

                if max_sim_runs is not None and sim_runs >= max_sim_runs:
                    stop_reason = StopReason.budget_exhausted
                    candidate_ok = False
                    candidate_errors = ["budget_exhausted"]
                    corner_results.append(
                        _corner_result(
                            corner_id=corner_id,
                            overrides=overrides,
                            metrics=None,
                            eval_result=None,
                            stage_map={},
                            warnings=[],
                            errors=candidate_errors,
                            guard_report=None,
                            recorder=recorder,
                        )
                    )
                    break

                corner_patch = _corner_patch(overrides)
                checks = []
                corner_guard: GuardReport | None = None
                corner_errors: list[str] = []
                stage_map: dict[str, str] = {}
                warnings: list[str] = []
                metrics_i: MetricsBundle | None = None
                eval_i: dict[str, Any] | None = None

                pre_check_res = self.patch_guard_op.run(
                    {
                        "circuit_ir": new_circuit_ir,
                        "param_space": param_space,
                        "patch": corner_patch,
                        "spec": spec,
                        "guard_cfg": guard_cfg,
                    },
                    ctx=None,
                )
                record_operator_result(recorder, pre_check_res)
                pre_check = pre_check_res.outputs["check"]
                checks.append(pre_check)
                if not pre_check.ok:
                    guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                    record_operator_result(recorder, guard_chain_res)
                    corner_guard = guard_chain_res.outputs["report"]
                    corner_errors = guard_failures(corner_guard)
                    candidate_ok = False
                    candidate_errors = corner_errors
                    corner_results.append(
                        _corner_result(
                            corner_id=corner_id,
                            overrides=overrides,
                            metrics=None,
                            eval_result=None,
                            stage_map=stage_map,
                            warnings=warnings,
                            errors=corner_errors,
                            guard_report=corner_guard,
                            recorder=recorder,
                        )
                    )
                    break

                try:
                    apply_res = self.patch_apply_op.run(
                        {"source": new_source, "param_space": param_space, "patch": corner_patch, **apply_opts},
                        ctx=None,
                    )
                    record_operator_result(recorder, apply_res)
                    apply_outputs = apply_res.outputs
                    corner_source = apply_outputs["source"]
                    corner_signature = apply_outputs["topology_signature"]
                except ValidationError as exc:
                    check_name = "topology_guard" if "Topology changed" in str(exc) else "patch_guard"
                    checks.append(
                        GuardCheck(
                            name=check_name,
                            ok=False,
                            severity="hard",
                            reasons=(str(exc),),
                        )
                    )
                    guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                    record_operator_result(recorder, guard_chain_res)
                    corner_guard = guard_chain_res.outputs["report"]
                    corner_errors = guard_failures(corner_guard)
                    candidate_ok = False
                    candidate_errors = corner_errors
                    corner_results.append(
                        _corner_result(
                            corner_id=corner_id,
                            overrides=overrides,
                            metrics=None,
                            eval_result=None,
                            stage_map=stage_map,
                            warnings=warnings,
                            errors=corner_errors,
                            guard_report=corner_guard,
                            recorder=recorder,
                        )
                    )
                    break

                topo_check_res = self.topology_guard_op.run(
                    {"signature_before": new_signature, "signature_after": corner_signature},
                    ctx=None,
                )
                record_operator_result(recorder, topo_check_res)
                topo_check = topo_check_res.outputs["check"]
                checks.append(topo_check)
                if not topo_check.ok:
                    guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                    record_operator_result(recorder, guard_chain_res)
                    corner_guard = guard_chain_res.outputs["report"]
                    corner_errors = guard_failures(corner_guard)
                    candidate_ok = False
                    candidate_errors = corner_errors
                    corner_results.append(
                        _corner_result(
                            corner_id=corner_id,
                            overrides=overrides,
                            metrics=None,
                            eval_result=None,
                            stage_map=stage_map,
                            warnings=warnings,
                            errors=corner_errors,
                            guard_report=corner_guard,
                            recorder=recorder,
                        )
                    )
                    break

                try:
                    measurement = _measure_corner_metrics(
                        source=corner_source,
                        corner_id=corner_id,
                        metric_groups=metric_groups,
                        ctx=ctx,
                        iter_idx=idx,
                        attempt_idx=0,
                        recorder=recorder,
                        manifest=manifest,
                        measure_fn=self.measure_fn,
                        deck_build_op=self.deck_build_op,
                        sim_run_op=self.sim_run_op,
                        metrics_op=self.metrics_op,
                    )
                    metrics_i = measurement.metrics
                    stage_map = measurement.stage_map
                    warnings = measurement.warnings
                    sim_runs += measurement.sim_runs
                    sim_runs_ok += measurement.sim_runs
                except (SimulationError, MetricError, ValidationError) as exc:
                    sim_runs += 1
                    sim_runs_failed += 1
                    checks.append(
                        GuardCheck(
                            name="behavior_guard",
                            ok=False,
                            severity="hard",
                            reasons=(str(exc),),
                            data={"error_type": type(exc).__name__},
                        )
                    )
                    guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                    record_operator_result(recorder, guard_chain_res)
                    corner_guard = guard_chain_res.outputs["report"]
                    corner_errors = guard_failures(corner_guard)
                    candidate_ok = False
                    candidate_errors = corner_errors
                    corner_results.append(
                        _corner_result(
                            corner_id=corner_id,
                            overrides=overrides,
                            metrics=None,
                            eval_result=None,
                            stage_map=stage_map,
                            warnings=warnings,
                            errors=corner_errors,
                            guard_report=corner_guard,
                            recorder=recorder,
                        )
                    )
                    break

                behavior_check_res = self.behavior_guard_op.run(
                    {"metrics": metrics_i, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg},
                    ctx=None,
                )
                record_operator_result(recorder, behavior_check_res)
                behavior_check = behavior_check_res.outputs["check"]
                checks.append(behavior_check)
                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                corner_guard = guard_chain_res.outputs["report"]
                corner_errors = guard_failures(corner_guard)

                if not corner_guard.ok:
                    candidate_ok = False
                    candidate_errors = corner_errors
                    corner_results.append(
                        _corner_result(
                            corner_id=corner_id,
                            overrides=overrides,
                            metrics=metrics_i,
                            eval_result=None,
                            stage_map=stage_map,
                            warnings=warnings,
                            errors=corner_errors,
                            guard_report=corner_guard,
                            recorder=recorder,
                        )
                    )
                    break

                eval_i = evaluate_metrics(spec, metrics_i)
                corner_metrics_bundle[corner_id] = metrics_i
                corner_results.append(
                    _corner_result(
                        corner_id=corner_id,
                        overrides=overrides,
                        metrics=metrics_i,
                        eval_result=eval_i,
                        stage_map=stage_map,
                        warnings=warnings,
                        errors=corner_errors,
                        guard_report=corner_guard,
                        recorder=recorder,
                    )
                )

            if not candidate_ok:
                history.append(
                    {
                        "iteration": idx,
                        "candidate": dict(candidate),
                        "patch": patch_to_dict(patch),
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {},
                        "score": float("inf"),
                        "all_pass": False,
                        "improved": False,
                        "objectives": [],
                        "sim_stages": {},
                        "warnings": [],
                        "errors": candidate_errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                        "corners": corner_results,
                        **_corner_summary_defaults(),
                    }
                )
                record_history_entry(recorder, history[-1])
                if stop_reason is StopReason.budget_exhausted:
                    break
                continue

            summary = aggregate_corner_results(corner_results)
            worst_entry = _pick_corner(corner_results, corner_id=summary.get("worst_corner_id"), worst=True)
            best_entry = _pick_corner(corner_results, worst=False)
            nominal_entry = _pick_corner(corner_results, corner_id="nominal", worst=False)
            nominal_metrics_bundle = corner_metrics_bundle.get("nominal")
            worst_metrics = worst_entry.get("metrics") if worst_entry else {}
            worst_objectives = worst_entry.get("objectives") if worst_entry else []
            worst_stage_map = worst_entry.get("sim_stages") if worst_entry else {}
            worst_warnings = worst_entry.get("warnings") if worst_entry else []
            worst_errors = worst_entry.get("errors") if worst_entry else []

            improved = summary["worst_score"] < best_score - 1e-12
            if improved:
                best_score = summary["worst_score"]
                best_metrics = nominal_metrics_bundle or baseline.metrics
                best_source = new_source
                best_all_pass = summary["pass_rate"] == 1.0
                best_iter = idx

            history.append(
                {
                    "iteration": idx,
                    "candidate": dict(candidate),
                    "patch": patch_to_dict(patch),
                    "signature_before": signature_before,
                    "signature_after": new_signature,
                    "metrics": worst_metrics,
                    "score": summary["worst_score"],
                    "all_pass": summary["pass_rate"] == 1.0,
                    "improved": improved,
                    "objectives": worst_objectives,
                    "sim_stages": worst_stage_map,
                    "warnings": worst_warnings,
                    "errors": worst_errors,
                    "guard": guard_report_to_dict(guard_report) if guard_report else None,
                    "attempts": attempts,
                    "corners": corner_results,
                    **summary,
                }
            )
            record_history_entry(recorder, history[-1])

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

            if stop_on_first_pass and summary["pass_rate"] == 1.0:
                stop_reason = StopReason.reached_target
                break

        if stop_reason is None:
            stop_reason = StopReason.max_iterations

        if recorder is not None:
            top_entries = top_k(candidate_entries, top_k_count)
            pareto_idx = pareto_front(candidate_losses)
            pareto_entries = [candidate_entries[i] for i in pareto_idx] if pareto_idx else []
            recorder.write_json("search/topk.json", top_entries)
            recorder.write_json("search/pareto.json", pareto_entries)
            report_lines = _build_report(
                cfg=cfg,
                corner_cfg=corner_cfg,
                baseline_eval=eval0,
                baseline_metrics=baseline.metrics,
                baseline_summary=summary0,
                candidate_entries=candidate_entries,
                pareto_entries=pareto_entries,
                sim_runs_failed=sim_runs_failed,
                param_value_errors=param_value_errors,
            )
            recorder.write_text("report.md", "\n".join(report_lines))

        recording_errors = finalize_run(
            recorder=recorder,
            manifest=manifest,
            best_source=best_source,
            best_metrics=best_metrics,
            history=history,
            stop_reason=stop_reason,
            best_score=best_score,
            best_iter=best_iter,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
        )

        return RunResult(
            best_source=best_source,
            best_metrics=best_metrics,
            history=history,
            stop_reason=stop_reason,
            notes={"best_score": best_score, "all_pass": best_all_pass, "recording_errors": recording_errors},
        )


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return "-"
    parts = [f"{name}={value}" for name, value in metrics.items()]
    return ", ".join(parts)


def _build_report(
    *,
    cfg: StrategyConfig,
    corner_cfg: Mapping[str, Any],
    baseline_eval: dict[str, Any],
    baseline_metrics: MetricsBundle,
    baseline_summary: Mapping[str, Any],
    candidate_entries: list[dict[str, Any]],
    pareto_entries: list[dict[str, Any]],
    sim_runs_failed: int,
    param_value_errors: list[str],
) -> list[str]:
    lines: list[str] = []
    lines.append("# Corner Search Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- max_iterations: {cfg.budget.max_iterations}")
    lines.append(f"- mode: {corner_cfg.get('mode', 'coordinate')}")
    lines.append(f"- levels: {corner_cfg.get('levels', 10)}")
    lines.append(f"- span_mul: {corner_cfg.get('span_mul', 10.0)}")
    lines.append(f"- scale: {corner_cfg.get('scale', 'log')}")
    lines.append(f"- corners: {corner_cfg.get('corners', 'oat')}")
    lines.append("")

    lines.append("## Baseline (Nominal)")
    lines.append(f"- score: {baseline_eval.get('score')}")
    lines.append(f"- all_pass: {baseline_eval.get('all_pass')}")
    lines.append(f"- worst_score: {baseline_summary.get('worst_score')}")
    lines.append(f"- pass_rate: {baseline_summary.get('pass_rate')}")
    lines.append(f"- worst_corner_id: {baseline_summary.get('worst_corner_id')}")
    for name, mv in baseline_metrics.values.items():
        lines.append(f"- metric {name}: {mv.value} {mv.unit or ''}".rstrip())
    if param_value_errors:
        lines.append("")
        lines.append("## Param Value Errors")
        for err in param_value_errors:
            lines.append(f"- {err}")

    lines.append("")
    lines.append("## Top-K Candidates (Worst Score)")
    for entry in top_k(candidate_entries, int(corner_cfg.get("top_k", 5))):
        lines.append(
            f"- iter {entry.get('iteration')}: worst_score={entry.get('worst_score')} "
            f"pass_rate={entry.get('pass_rate')} worst_corner_id={entry.get('worst_corner_id')} "
            f"nominal={_format_metrics(entry.get('nominal_metrics') or {})} "
            f"worst={_format_metrics(entry.get('worst_metrics') or {})} "
            f"best={_format_metrics(entry.get('best_metrics') or {})}"
        )

    lines.append("")
    lines.append("## Pareto Front (Robust Losses)")
    for entry in pareto_entries:
        lines.append(
            f"- iter {entry.get('iteration')}: losses={entry.get('robust_losses')} "
            f"worst_score={entry.get('worst_score')} worst_corner_id={entry.get('worst_corner_id')} "
            f"nominal={_format_metrics(entry.get('nominal_metrics') or {})} "
            f"worst={_format_metrics(entry.get('worst_metrics') or {})} "
            f"best={_format_metrics(entry.get('best_metrics') or {})}"
        )

    lines.append("")
    lines.append("## Failures")
    lines.append(f"- sim_runs_failed: {sim_runs_failed}")

    lines.append("")
    lines.append("## Files")
    lines.append("- search/corner_set.json")
    lines.append("- search/topk.json")
    lines.append("- search/pareto.json")
    lines.append("- history/iterations.jsonl")
    lines.append("- best/best.sp")
    lines.append("- best/best_metrics.json")
    return lines
