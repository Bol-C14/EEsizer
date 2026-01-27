from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ...analysis.objective_eval import evaluate_objectives
from ...contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamSpace, Patch, SimPlan
from ...contracts.enums import SimKind
from ...contracts.errors import MetricError, SimulationError, ValidationError
from ...contracts.guards import GuardCheck, GuardReport
from ...runtime.recorder import RunRecorder
from ...runtime.recording_utils import guard_failures, record_operator_result
from .sim_plan import merge_metrics, sim_plan_for_kind


MeasureFn = Callable[[CircuitSource, int], MetricsBundle]


@dataclass(frozen=True)
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
    sim_plan: SimPlan | None = None,
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
        plan = sim_plan if sim_plan is not None else sim_plan_for_kind(kind)
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


@dataclass(frozen=True)
class AttemptOperators:
    patch_guard_op: Any
    patch_apply_op: Any
    topology_guard_op: Any
    behavior_guard_op: Any
    guard_chain_op: Any
    formal_guard_op: Any | None
    deck_build_op: Any
    sim_run_op: Any
    metrics_op: Any


@dataclass
class AttemptResult:
    attempt: int
    patch: Patch
    guard_report: GuardReport | None
    guard_failures: list[str]
    stage_map: dict[str, str]
    warnings: list[str]
    metrics: MetricsBundle | None
    new_source: CircuitSource
    new_signature: str
    new_circuit_ir: Any
    sim_runs: int
    sim_runs_ok: int
    sim_runs_failed: int
    success: bool


def _run_guard_chain(guard_chain_op: Any, checks: list[GuardCheck], recorder: RunRecorder | None) -> GuardReport:
    guard_chain_res = guard_chain_op.run({"checks": checks}, ctx=None)
    record_operator_result(recorder, guard_chain_res)
    return guard_chain_res.outputs["report"]


def run_attempt(
    *,
    iter_idx: int,
    attempt: int,
    patch: Patch,
    cur_source: CircuitSource,
    cur_signature: str,
    circuit_ir: Any,
    param_space: ParamSpace,
    spec: CircuitSpec,
    guard_cfg: Mapping[str, Any],
    apply_opts: Mapping[str, Any],
    metric_groups: Mapping[SimKind, list[str]],
    ctx: Any,
    recorder: RunRecorder | None,
    manifest: Any,
    measure_fn: MeasureFn | None,
    ops: AttemptOperators,
    sim_plan: SimPlan | None = None,
    stage_tag: str | None = None,
) -> AttemptResult:
    new_source = cur_source
    new_signature = cur_signature
    new_circuit_ir = circuit_ir
    metrics = None
    stage_map: dict[str, str] = {}
    warnings: list[str] = []
    sim_runs = 0
    sim_runs_ok = 0
    sim_runs_failed = 0

    checks: list[GuardCheck] = []
    pre_check_res = ops.patch_guard_op.run(
        {"circuit_ir": circuit_ir, "param_space": param_space, "patch": patch, "spec": spec, "guard_cfg": guard_cfg},
        ctx=None,
    )
    record_operator_result(recorder, pre_check_res)
    pre_check = pre_check_res.outputs["check"]
    checks.append(pre_check)
    if not pre_check.ok:
        guard_report = _run_guard_chain(ops.guard_chain_op, checks, recorder)
        failures = guard_failures(guard_report)
        return AttemptResult(
            attempt=attempt,
            patch=patch,
            guard_report=guard_report,
            guard_failures=failures,
            stage_map=stage_map,
            warnings=warnings,
            metrics=metrics,
            new_source=new_source,
            new_signature=new_signature,
            new_circuit_ir=new_circuit_ir,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
            success=False,
        )

    try:
        apply_res = ops.patch_apply_op.run({"source": cur_source, "param_space": param_space, "patch": patch, **apply_opts}, ctx=None)
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
        guard_report = _run_guard_chain(ops.guard_chain_op, checks, recorder)
        failures = guard_failures(guard_report)
        return AttemptResult(
            attempt=attempt,
            patch=patch,
            guard_report=guard_report,
            guard_failures=failures,
            stage_map=stage_map,
            warnings=warnings,
            metrics=metrics,
            new_source=new_source,
            new_signature=new_signature,
            new_circuit_ir=new_circuit_ir,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
            success=False,
        )

    topo_check_res = ops.topology_guard_op.run({"signature_before": cur_signature, "signature_after": new_signature}, ctx=None)
    record_operator_result(recorder, topo_check_res)
    topo_check = topo_check_res.outputs["check"]
    checks.append(topo_check)
    if not topo_check.ok:
        guard_report = _run_guard_chain(ops.guard_chain_op, checks, recorder)
        failures = guard_failures(guard_report)
        return AttemptResult(
            attempt=attempt,
            patch=patch,
            guard_report=guard_report,
            guard_failures=failures,
            stage_map=stage_map,
            warnings=warnings,
            metrics=metrics,
            new_source=new_source,
            new_signature=new_signature,
            new_circuit_ir=new_circuit_ir,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
            success=False,
        )

    if ops.formal_guard_op is not None:
        formal_check_res = ops.formal_guard_op.run({"source": new_source, "circuit_ir": new_circuit_ir, "spec": spec}, ctx=None)
        record_operator_result(recorder, formal_check_res)
        formal_check = formal_check_res.outputs["check"]
        checks.append(formal_check)

    measurement = measure_metrics(
        source=new_source,
        metric_groups=metric_groups,
        ctx=ctx,
        iter_idx=iter_idx,
        attempt_idx=attempt,
        recorder=recorder,
        manifest=manifest,
        measure_fn=measure_fn,
        deck_build_op=ops.deck_build_op,
        sim_run_op=ops.sim_run_op,
        metrics_op=ops.metrics_op,
        sim_plan=sim_plan,
        stage_tag=stage_tag,
    )
    sim_runs = measurement.sim_runs
    sim_runs_ok = measurement.sim_runs_ok
    sim_runs_failed = measurement.sim_runs_failed

    if not measurement.ok:
        error = measurement.error or SimulationError("measurement_failed")
        checks.append(
            GuardCheck(
                name="behavior_guard",
                ok=False,
                severity="hard",
                reasons=(str(error),),
                data={"error_type": type(error).__name__},
            )
        )
        guard_report = _run_guard_chain(ops.guard_chain_op, checks, recorder)
        failures = guard_failures(guard_report)
        return AttemptResult(
            attempt=attempt,
            patch=patch,
            guard_report=guard_report,
            guard_failures=failures,
            stage_map=stage_map,
            warnings=warnings,
            metrics=metrics,
            new_source=new_source,
            new_signature=new_signature,
            new_circuit_ir=new_circuit_ir,
            sim_runs=sim_runs,
            sim_runs_ok=sim_runs_ok,
            sim_runs_failed=sim_runs_failed,
            success=False,
        )

    metrics = measurement.metrics
    stage_map = measurement.stage_map
    warnings = measurement.warnings

    behavior_check_res = ops.behavior_guard_op.run({"metrics": metrics, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg}, ctx=None)
    record_operator_result(recorder, behavior_check_res)
    behavior_check = behavior_check_res.outputs["check"]
    checks.append(behavior_check)

    guard_report = _run_guard_chain(ops.guard_chain_op, checks, recorder)
    failures = guard_failures(guard_report)
    return AttemptResult(
        attempt=attempt,
        patch=patch,
        guard_report=guard_report,
        guard_failures=failures,
        stage_map=stage_map,
        warnings=warnings,
        metrics=metrics,
        new_source=new_source,
        new_signature=new_signature,
        new_circuit_ir=new_circuit_ir,
        sim_runs=sim_runs,
        sim_runs_ok=sim_runs_ok,
        sim_runs_failed=sim_runs_failed,
        success=guard_report.ok,
    )


def evaluate_metrics(spec: CircuitSpec, metrics: MetricsBundle) -> dict[str, Any]:
    return evaluate_objectives(spec, metrics)

