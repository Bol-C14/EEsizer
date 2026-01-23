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
            history.append(
                {
                    "iteration": 0,
                    "candidate": None,
                    "patch": None,
                    "signature_before": signature,
                    "signature_after": signature,
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
            record_history_entry(recorder, history[-1])
            recording_errors = finalize_run(
                recorder=recorder,
                manifest=manifest,
                best_source=source,
                best_metrics=MetricsBundle(),
                history=history,
                stop_reason=baseline.stop_reason,
                best_score=float("inf"),
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
                    "best_score": float("inf"),
                    "all_pass": False,
                    "recording_errors": recording_errors,
                },
            )

        eval0 = evaluate_metrics(spec, baseline.metrics)
        best_source = source
        best_metrics = baseline.metrics
        best_score = eval0["score"]
        best_all_pass = eval0["all_pass"]
        best_iter = 0
        stop_reason: StopReason | None = StopReason.reached_target if best_all_pass else None

        sim_runs = baseline.sim_runs
        sim_runs_ok = baseline.sim_runs_ok
        sim_runs_failed = baseline.sim_runs_failed

        history.append(
            {
                "iteration": 0,
                "candidate": None,
                "patch": None,
                "signature_before": signature,
                "signature_after": signature,
                "metrics": {k: v.value for k, v in baseline.metrics.values.items()},
                "score": best_score,
                "all_pass": best_all_pass,
                "improved": False,
                "objectives": eval0["per_objective"],
                "sim_stages": baseline.stage_map,
                "warnings": baseline.warnings,
                "errors": baseline.errors,
                "guard": guard_report_to_dict(baseline.guard_report) if baseline.guard_report else None,
                "attempts": baseline.attempts,
            }
        )
        record_history_entry(recorder, history[-1])

        param_ids = [p.param_id for p in param_space.params if not p.frozen]
        cfg_param_ids = grid_cfg.get("param_ids")
        if isinstance(cfg_param_ids, (list, tuple)):
            param_ids = [str(pid).lower() for pid in cfg_param_ids]

        baseline_values, param_value_errors = extract_param_values(circuit_ir, param_ids=param_ids)
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

        validation_opts = {
            "wl_ratio_min": guard_cfg.get("wl_ratio_min"),
            "max_mul_factor": guard_cfg.get("max_mul_factor", 10.0),
        }
        apply_opts = {
            "include_paths": cfg.notes.get("include_paths", True),
            "max_lines": cfg.notes.get("max_lines", 50000),
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

        for idx, candidate in enumerate(candidates, start=1):
            if max_sim_runs is not None and sim_runs >= max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            patch_ops = [
                PatchOp(param=pid, op=PatchOpType.set, value=val, why="grid_search")
                for pid, val in candidate.items()
            ]
            patch = Patch(ops=tuple(patch_ops))
            signature_before = signature
            attempt_result = run_attempt(
                iter_idx=idx,
                attempt=0,
                patch=patch,
                cur_source=source,
                cur_signature=signature,
                circuit_ir=circuit_ir,
                param_space=param_space,
                spec=spec,
                guard_cfg=guard_cfg,
                apply_opts=apply_opts,
                metric_groups=metric_groups,
                ctx=ctx,
                recorder=recorder,
                manifest=manifest,
                measure_fn=self.measure_fn,
                ops=attempt_ops,
            )
            sim_runs += attempt_result.sim_runs
            sim_runs_ok += attempt_result.sim_runs_ok
            sim_runs_failed += attempt_result.sim_runs_failed
            metrics_i = attempt_result.metrics
            stage_map_i = attempt_result.stage_map
            warnings_i = attempt_result.warnings
            guard_report = attempt_result.guard_report
            errors = attempt_result.guard_failures
            attempts = [attempt_record(0, patch, guard_report, stage_map_i, warnings_i)]
            new_source = attempt_result.new_source
            new_signature = attempt_result.new_signature

            if not attempt_result.success or metrics_i is None:
                metrics_payload = {k: v.value for k, v in metrics_i.values.items()} if metrics_i else {}
                history.append(
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
                record_history_entry(recorder, history[-1])
                continue

            eval_i = evaluate_metrics(spec, metrics_i)
            improved = eval_i["score"] < best_score - 1e-12
            if improved:
                best_score = eval_i["score"]
                best_metrics = metrics_i
                best_source = new_source
                best_all_pass = eval_i["all_pass"]
                best_iter = idx

            history.append(
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
            record_history_entry(recorder, history[-1])

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

            if stop_on_first_pass and eval_i["all_pass"]:
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
                grid_cfg=grid_cfg,
                baseline_eval=eval0,
                baseline_metrics=baseline.metrics,
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
