from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import CircuitSource, CircuitSpec, ParamSpace, Patch
from ..contracts.enums import SimKind
from ..contracts.errors import SimulationError, ValidationError
from ..contracts.guards import GuardCheck, GuardReport
from ..runtime.recorder import RunRecorder
from ..runtime.recording_utils import guard_failures, record_operator_result
from .patch_loop.evaluate import MeasureFn, measure_metrics
from .patch_loop.state import AttemptResult


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
        apply_res = ops.patch_apply_op.run(
            {"source": cur_source, "param_space": param_space, "patch": patch, **apply_opts},
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

    topo_check_res = ops.topology_guard_op.run(
        {"signature_before": cur_signature, "signature_after": new_signature},
        ctx=None,
    )
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
        formal_check_res = ops.formal_guard_op.run(
            {"source": new_source, "circuit_ir": new_circuit_ir, "spec": spec},
            ctx=None,
        )
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

    behavior_check_res = ops.behavior_guard_op.run(
        {"metrics": metrics, "spec": spec, "stage_map": stage_map, "guard_cfg": guard_cfg},
        ctx=None,
    )
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
