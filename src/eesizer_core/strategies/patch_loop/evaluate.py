from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ...contracts import CircuitSource, CircuitSpec, MetricsBundle
from ...contracts.enums import SimKind, StopReason
from ...contracts.errors import MetricError, SimulationError, ValidationError
from ...contracts.guards import GuardCheck, GuardReport
from ...runtime.recorder import RunRecorder
from ...runtime.recording_utils import attempt_record, guard_failures, record_operator_result
from ...analysis.objective_eval import evaluate_objectives
from .planning import merge_metrics, sim_plan_for_kind
from .state import BaselineResult

MeasureFn = Callable[[CircuitSource, int], MetricsBundle]


@dataclass
class MeasurementResult:
    metrics: MetricsBundle
    stage_map: dict[str, str]
    warnings: list[str]
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int
    ok: bool
    error: Exception | None = None


def _failure_result(
    error: Exception,
    *,
    sim_runs: int,
    sim_runs_ok: int,
    sim_runs_failed: int,
    stage_map: dict[str, str],
    warnings: list[str],
) -> MeasurementResult:
    return MeasurementResult(
        metrics=MetricsBundle(),
        stage_map=stage_map,
        warnings=warnings,
        sim_runs=sim_runs,
        sim_runs_ok=sim_runs_ok,
        sim_runs_failed=sim_runs_failed,
        ok=False,
        error=error,
    )


def measure_metrics(
    *,
    source: CircuitSource,
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
    stage_tag: str | None = None,
) -> MeasurementResult:
    if measure_fn is not None:
        return MeasurementResult(
            metrics=measure_fn(source, iter_idx),
            stage_map={},
            warnings=[],
            sim_runs=0,
            sim_runs_ok=0,
            sim_runs_failed=0,
            ok=True,
        )

    bundles: list[MetricsBundle] = []
    stage_map: dict[str, str] = {}
    warnings: list[str] = []
    sim_runs = 0
    sim_runs_ok = 0
    sim_runs_failed = 0

    for kind, names in metric_groups.items():
        plan = sim_plan_for_kind(kind)
        try:
            deck_res = deck_build_op.run({"circuit_source": source, "sim_plan": plan, "sim_kind": kind}, ctx=None)
        except (SimulationError, MetricError, ValidationError) as exc:
            return _failure_result(
                exc,
                sim_runs=sim_runs,
                sim_runs_ok=sim_runs_ok,
                sim_runs_failed=sim_runs_failed,
                stage_map=stage_map,
                warnings=warnings,
            )
        record_operator_result(recorder, deck_res)
        deck = deck_res.outputs["deck"]
        tag = f"_{stage_tag}" if stage_tag else ""
        stage_name = f"{kind.value}{tag}_i{iter_idx:03d}_a{attempt_idx:02d}"
        sim_runs += 1
        try:
            run_res = sim_run_op.run({"deck": deck, "stage": stage_name}, ctx)
        except (SimulationError, MetricError, ValidationError) as exc:
            sim_runs_failed += 1
            return _failure_result(
                exc,
                sim_runs=sim_runs,
                sim_runs_ok=sim_runs_ok,
                sim_runs_failed=sim_runs_failed,
                stage_map=stage_map,
                warnings=warnings,
            )
        record_operator_result(recorder, run_res)
        if manifest is not None:
            version = run_res.provenance.notes.get("ngspice_version")
            path = run_res.provenance.notes.get("ngspice_path")
            if version:
                manifest.environment.setdefault("ngspice_version", version)
            if path:
                manifest.environment.setdefault("ngspice_path", path)
        raw = run_res.outputs["raw_data"]
        try:
            metrics_res = metrics_op.run({"raw_data": raw, "metric_names": names}, ctx=None)
        except (SimulationError, MetricError, ValidationError) as exc:
            sim_runs_failed += 1
            return _failure_result(
                exc,
                sim_runs=sim_runs,
                sim_runs_ok=sim_runs_ok,
                sim_runs_failed=sim_runs_failed,
                stage_map=stage_map,
                warnings=warnings,
            )
        record_operator_result(recorder, metrics_res)
        bundles.append(metrics_res.outputs["metrics"])
        stage_map[kind.value] = str(raw.run_dir)
        warnings.extend(deck_res.warnings)
        warnings.extend(run_res.warnings)
        sim_runs_ok += 1

    return MeasurementResult(
        metrics=merge_metrics(bundles),
        stage_map=stage_map,
        warnings=warnings,
        sim_runs=sim_runs,
        sim_runs_ok=sim_runs_ok,
        sim_runs_failed=sim_runs_failed,
        ok=True,
    )


def evaluate_metrics(spec: CircuitSpec, metrics: MetricsBundle) -> dict[str, Any]:
    return evaluate_objectives(spec, metrics)


def run_baseline(
    *,
    source: CircuitSource,
    spec: CircuitSpec,
    metric_groups: Mapping[SimKind, list[str]],
    ctx: Any,
    guard_cfg: Mapping[str, Any],
    max_retries: int,
    max_sim_runs: int | None,
    recorder: RunRecorder | None,
    manifest: Any,
    measure_fn: MeasureFn | None,
    deck_build_op: Any,
    sim_run_op: Any,
    metrics_op: Any,
    behavior_guard_op: Any,
    guard_chain_op: Any,
) -> BaselineResult:
    attempts: list[dict[str, Any]] = []
    guard_report: GuardReport | None = None
    errors: list[str] = []
    metrics = MetricsBundle()
    stage_map: dict[str, str] = {}
    warnings: list[str] = []
    sim_runs = 0
    sim_runs_ok = 0
    sim_runs_failed = 0
    success = False
    stop_reason: StopReason | None = None

    for attempt in range(max_retries + 1):
        if max_sim_runs is not None and sim_runs >= max_sim_runs:
            stop_reason = StopReason.budget_exhausted
            break
        measurement = measure_metrics(
            source=source,
            metric_groups=metric_groups,
            ctx=ctx,
            iter_idx=0,
            attempt_idx=attempt,
            recorder=recorder,
            manifest=manifest,
            measure_fn=measure_fn,
            deck_build_op=deck_build_op,
            sim_run_op=sim_run_op,
            metrics_op=metrics_op,
        )
        metrics = measurement.metrics
        stage_map = measurement.stage_map
        warnings = measurement.warnings
        sim_runs += measurement.sim_runs
        sim_runs_ok += measurement.sim_runs_ok
        sim_runs_failed += measurement.sim_runs_failed

        if not measurement.ok:
            error = measurement.error or SimulationError("measurement_failed")
            check = GuardCheck(
                name="behavior_guard",
                ok=False,
                severity="hard",
                reasons=(str(error),),
                data={"error_type": type(error).__name__},
            )
            guard_chain_res = guard_chain_op.run({"checks": [check]}, ctx=None)
            record_operator_result(recorder, guard_chain_res)
            guard_report = guard_chain_res.outputs["report"]
            errors = guard_failures(guard_report)
            attempts.append(attempt_record(attempt, None, guard_report))
            if attempt >= max_retries:
                break
            continue

        behavior_check_res = behavior_guard_op.run(
            {"metrics": metrics, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg},
            ctx=None,
        )
        record_operator_result(recorder, behavior_check_res)
        behavior_check = behavior_check_res.outputs["check"]
        guard_chain_res = guard_chain_op.run({"checks": [behavior_check]}, ctx=None)
        record_operator_result(recorder, guard_chain_res)
        guard_report = guard_chain_res.outputs["report"]
        attempts.append(attempt_record(attempt, None, guard_report, stage_map, warnings))
        errors = guard_failures(guard_report)

        if not guard_report.ok:
            if attempt >= max_retries:
                break
            continue

        success = True
        break

    if not success and stop_reason is None:
        stop_reason = StopReason.guard_failed

    return BaselineResult(
        success=success,
        metrics=metrics,
        guard_report=guard_report,
        attempts=attempts,
        errors=errors,
        stage_map=stage_map,
        warnings=warnings,
        stop_reason=stop_reason,
        sim_runs=sim_runs,
        sim_runs_ok=sim_runs_ok,
        sim_runs_failed=sim_runs_failed,
    )
