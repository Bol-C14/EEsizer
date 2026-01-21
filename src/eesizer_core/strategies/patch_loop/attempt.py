from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json

from ...contracts import CircuitSource, CircuitSpec, ParamSpace, Patch
from ...contracts.enums import SimKind
from ...contracts.errors import MetricError, SimulationError, ValidationError
from ...contracts.guards import GuardCheck, GuardReport
from ...contracts.policy import Observation
from ...runtime.recorder import RunRecorder
from ...runtime.recording_utils import guard_failures, record_operator_result
from .evaluate import MeasureFn, measure_metrics
from .state import AttemptResult


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


def _llm_patch_payload(patch: Patch) -> dict[str, Any]:
    return {
        "patch": [
            {"param": op.param, "op": getattr(op.op, "value", op.op), "value": op.value, "why": getattr(op, "why", "")}
            for op in patch.ops
        ],
        "stop": patch.stop,
        "notes": patch.notes,
    }


def _llm_stage_name(obs: Observation, retry_idx: int) -> str:
    attempt = obs.notes.get("attempt", 0)
    base = f"llm/llm_i{obs.iteration:03d}_a{attempt:02d}"
    if retry_idx > 0:
        return f"{base}_r{retry_idx:02d}"
    return base


def _write_llm_artifact(ctx: Any, stage: str, filename: str, payload: str) -> None:
    if ctx is None or not hasattr(ctx, "run_dir"):
        return
    stage_dir = Path(ctx.run_dir()) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / filename).write_text(payload, encoding="utf-8")


def is_llm_policy(policy: Any) -> bool:
    return hasattr(policy, "build_request") and hasattr(policy, "parse_response")


def _propose_llm_patch(
    policy: Any,
    llm_call_op: Any,
    obs: Observation,
    ctx: Any,
) -> Patch:
    last_error: str | None = None
    max_retries = int(getattr(policy, "max_retries", 0))

    for retry in range(max_retries + 1):
        request_payload, stop_reason = policy.build_request(obs, last_error=last_error)
        if stop_reason:
            return Patch(stop=True, notes=stop_reason)
        if not isinstance(request_payload, dict):
            return Patch(stop=True, notes="llm_request_missing")

        stage = _llm_stage_name(obs, retry)
        inputs = {"request": request_payload, "stage": stage}
        provider = request_payload.get("config", {}).get("provider", getattr(policy, "provider", None))
        if provider == "mock" and hasattr(policy, "mock_response"):
            prompt = request_payload.get("user", "")
            inputs["mock_response"] = policy.mock_response(prompt, obs)
        try:
            llm_result = llm_call_op.run(inputs, ctx)
        except Exception as exc:
            _write_llm_artifact(ctx, stage, "call_error.txt", str(exc))
            return Patch(stop=True, notes="llm_call_failed")

        response_text = llm_result.outputs.get("response_text", "")
        try:
            patch = policy.parse_response(response_text, obs)
        except Exception as exc:
            last_error = str(exc)
            _write_llm_artifact(ctx, stage, "parse_error.txt", last_error)
            continue

        _write_llm_artifact(
            ctx,
            stage,
            "parsed_patch.json",
            json.dumps(_llm_patch_payload(patch), indent=2, sort_keys=True),
        )
        return patch

    return Patch(stop=True, notes="llm_parse_failed")


def propose_patch(policy: Any, llm_call_op: Any, obs: Observation, ctx: Any) -> Patch:
    if is_llm_policy(policy):
        return _propose_llm_patch(policy, llm_call_op, obs, ctx)
    return policy.propose(obs, ctx)


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

    try:
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
        )
    except (SimulationError, MetricError) as exc:
        sim_runs = 1
        sim_runs_failed = 1
        checks.append(
            GuardCheck(
                name="behavior_guard",
                ok=False,
                severity="hard",
                reasons=(str(exc),),
                data={"error_type": type(exc).__name__},
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
    sim_runs = measurement.sim_runs
    sim_runs_ok = measurement.sim_runs

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
