from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import (
    CircuitSource,
    CircuitSpec,
    MetricsBundle,
    RunResult,
    SimPlan,
    SimRequest,
    StrategyConfig,
)
from ..contracts.enums import SimKind, StopReason
from ..contracts.errors import MetricError, SimulationError
from ..contracts.guards import GuardCheck, GuardReport
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..contracts.strategy import Strategy
from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ..metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ..metrics.aliases import canonicalize_metrics
from ..metrics.reporting import format_metric_line, metric_definition_lines
from ..operators.guards import BehaviorGuardOperator, GuardChainOperator
from ..operators.netlist import TopologySignatureOperator
from ..runtime.recorder import RunRecorder
from ..runtime.recording_utils import (
    attempt_record,
    finalize_run,
    guard_failures,
    guard_report_to_dict,
    metrics_to_dict,
    param_space_to_dict,
    record_history_entry,
    record_operator_result,
    spec_to_dict,
    strategy_cfg_to_dict,
)
from ..sim import DeckBuildOperator, NgspiceRunOperator
from ..analysis.objective_eval import evaluate_objectives
from .patch_loop.planning import group_metric_names_by_kind, merge_metrics, sim_plan_for_kind


def _extract_sim_plan(notes: Mapping[str, Any]) -> SimPlan | None:
    raw = notes.get("sim_plan")
    if isinstance(raw, SimPlan):
        return raw
    if isinstance(raw, Mapping):
        sims_raw = raw.get("sims")
        if not isinstance(sims_raw, list) or not sims_raw:
            return None
        sims: list[SimRequest] = []
        for item in sims_raw:
            if not isinstance(item, Mapping):
                return None
            kind = item.get("kind")
            params = item.get("params", {})
            if not isinstance(kind, str) or not isinstance(params, Mapping):
                return None
            try:
                sim_kind = SimKind(kind)
            except ValueError:
                return None
            sims.append(SimRequest(kind=sim_kind, params=dict(params)))
        return SimPlan(sims=tuple(sims))
    return None


def _build_baseline_report(spec: CircuitSpec, metrics: MetricsBundle, eval0: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("# Baseline Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- score: {eval0.get('score')}")
    lines.append(f"- all_pass: {eval0.get('all_pass')}")
    lines.append("")

    metric_names = [obj.metric for obj in spec.objectives]
    lines.extend(metric_definition_lines(metric_names))
    if metric_names:
        lines.append("")

    lines.append("## Metrics")
    for name, mv in metrics.values.items():
        lines.append(format_metric_line(name, mv))
    return lines


@dataclass
class NoOptBaselineStrategy(Strategy):
    """Run a single baseline simulation without any patching."""

    name: str = "baseline_noopt"
    version: str = "0.1.0"

    signature_op: Any = None
    deck_build_op: Any = None
    sim_run_op: Any = None
    metrics_op: Any = None
    behavior_guard_op: Any = None
    guard_chain_op: Any = None
    registry: MetricRegistry | None = None

    def __post_init__(self) -> None:
        if self.signature_op is None:
            self.signature_op = TopologySignatureOperator()
        if self.deck_build_op is None:
            self.deck_build_op = DeckBuildOperator()
        if self.sim_run_op is None:
            self.sim_run_op = NgspiceRunOperator()
        if self.metrics_op is None:
            self.metrics_op = ComputeMetricsOperator()
        if self.behavior_guard_op is None:
            self.behavior_guard_op = BehaviorGuardOperator()
        if self.guard_chain_op is None:
            self.guard_chain_op = GuardChainOperator()
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
            manifest.files.setdefault("report.md", "report.md")
            if recorder is not None:
                recorder.write_input("source.sp", source.text)
                recorder.write_input("spec.json", spec_payload)
                recorder.write_input("param_space.json", param_payload)
                recorder.write_input("cfg.json", cfg_payload)
                recorder.write_input("signature.txt", signature)

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)
        sim_plan = _extract_sim_plan(spec.notes) or _extract_sim_plan(cfg.notes)

        bundles: list[MetricsBundle] = []
        stage_map: dict[str, str] = {}
        warnings: list[str] = []
        errors: list[str] = []
        guard_report: GuardReport | None = None
        sim_runs = 0
        sim_runs_ok = 0
        sim_runs_failed = 0

        try:
            for kind, names in metric_groups.items():
                plan = sim_plan if sim_plan is not None else sim_plan_for_kind(kind)
                deck_res = self.deck_build_op.run(
                    {"circuit_source": source, "sim_plan": plan, "sim_kind": kind},
                    ctx=None,
                )
                record_operator_result(recorder, deck_res)
                deck = deck_res.outputs["deck"]
                stage_name = f"{kind.value}_i000_a00"
                run_res = self.sim_run_op.run({"deck": deck, "stage": stage_name}, ctx)
                record_operator_result(recorder, run_res)
                if manifest is not None:
                    version = run_res.provenance.notes.get("ngspice_version")
                    path = run_res.provenance.notes.get("ngspice_path")
                    if version:
                        manifest.environment.setdefault("ngspice_version", version)
                    if path:
                        manifest.environment.setdefault("ngspice_path", path)
                raw = run_res.outputs["raw_data"]
                metrics_res = self.metrics_op.run({"raw_data": raw, "metric_names": names}, ctx=None)
                record_operator_result(recorder, metrics_res)
                bundles.append(metrics_res.outputs["metrics"])
                stage_map[kind.value] = str(raw.run_dir)
                warnings.extend(deck_res.warnings)
                warnings.extend(run_res.warnings)
                sim_runs += 1
                sim_runs_ok += 1
        except (SimulationError, MetricError) as exc:
            sim_runs += 1
            sim_runs_failed += 1
            check = GuardCheck(
                name="behavior_guard",
                ok=False,
                severity="hard",
                reasons=(str(exc),),
                data={"error_type": type(exc).__name__},
            )
            guard_res = self.guard_chain_op.run({"checks": [check]}, ctx=None)
            record_operator_result(recorder, guard_res)
            guard_report = guard_res.outputs["report"]
            errors = guard_failures(guard_report)

        metrics = merge_metrics(bundles) if bundles else MetricsBundle()
        if guard_report is None:
            behavior_res = self.behavior_guard_op.run(
                {"metrics": metrics, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg},
                ctx=None,
            )
            record_operator_result(recorder, behavior_res)
            behavior_check = behavior_res.outputs["check"]
            guard_res = self.guard_chain_op.run({"checks": [behavior_check]}, ctx=None)
            record_operator_result(recorder, guard_res)
            guard_report = guard_res.outputs["report"]
            errors = guard_failures(guard_report)

        eval0 = evaluate_objectives(spec, metrics)
        history.append(
            {
                "iteration": 0,
                "patch": None,
                "signature_before": signature,
                "signature_after": signature,
                "metrics": {k: v.value for k, v in metrics.values.items()},
                "score": eval0["score"],
                "all_pass": eval0["all_pass"],
                "improved": False,
                "objectives": eval0["per_objective"],
                "sim_stages": stage_map,
                "warnings": warnings,
                "errors": errors,
                "guard": guard_report_to_dict(guard_report) if guard_report else None,
                "attempts": [attempt_record(0, None, guard_report, stage_map, warnings)],
            }
        )
        record_history_entry(recorder, history[-1])
        if recorder is not None:
            report_lines = _build_baseline_report(spec, metrics, eval0)
            recorder.write_text("report.md", "\n".join(report_lines))

        recording_errors = finalize_run(
            recorder=recorder,
            manifest=manifest,
            best_source=source,
            best_metrics=metrics,
            history=history,
            stop_reason=StopReason.baseline_noopt,
            best_score=eval0["score"],
            best_iter=0,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
            best_metrics_payload=canonicalize_metrics(metrics_to_dict(metrics)),
        )

        return RunResult(
            best_source=source,
            best_metrics=metrics,
            history=history,
            stop_reason=StopReason.baseline_noopt,
            notes={
                "best_score": eval0["score"],
                "all_pass": eval0["all_pass"],
                "recording_errors": recording_errors,
            },
        )
