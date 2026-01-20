from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional
import json

from ..contracts import (
    CircuitSource,
    CircuitSpec,
    MetricsBundle,
    ParamSpace,
    Patch,
    RunResult,
    SimPlan,
    SimRequest,
    StrategyConfig,
)
from ..contracts.enums import SimKind, StopReason
from ..contracts.errors import MetricError, SimulationError, ValidationError
from ..contracts.guards import GuardCheck, GuardReport
from ..contracts.policy import Observation, Policy
from ..contracts.strategy import Strategy
from ..contracts.provenance import stable_hash_json, stable_hash_str
from ..metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ..operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ..operators.guards import (
    PatchGuardOperator,
    TopologyGuardOperator,
    BehaviorGuardOperator,
    GuardChainOperator,
)
from ..operators.llm import LLMCallOperator
from ..sim import DeckBuildOperator, NgspiceRunOperator
from ..strategies.objective_eval import evaluate_objectives
from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ..domain.spice.patching import extract_param_values
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


def _is_llm_policy(policy: Any) -> bool:
    return hasattr(policy, "build_request") and hasattr(policy, "parse_response")


def _propose_llm_patch(
    policy: Any,
    llm_call_op: Any,
    obs: Observation,
    ctx: Any,
) -> Patch:
    last_error: Optional[str] = None
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


def _group_metric_names_by_kind(registry: MetricRegistry, metric_names: Iterable[str]) -> dict[SimKind, list[str]]:
    grouped: dict[SimKind, list[str]] = {}
    specs = registry.resolve(metric_names)
    for spec in specs:
        grouped.setdefault(spec.requires_kind, []).append(spec.name)
    return grouped


def _sim_plan_for_kind(kind: SimKind) -> SimPlan:
    return SimPlan(sims=(SimRequest(kind=kind, params={}),))


def _merge_metrics(bundles: list[MetricsBundle]) -> MetricsBundle:
    out = MetricsBundle()
    for b in bundles:
        out.values.update(b.values)
    return out


def _make_observation(
    spec: CircuitSpec,
    source: CircuitSource,
    param_space: ParamSpace,
    metrics: MetricsBundle,
    iteration: int,
    history: list[dict[str, Any]],
    history_tail_k: int,
    notes: Mapping[str, Any],
) -> Observation:
    tail = history[-history_tail_k:] if history_tail_k > 0 else []
    return Observation(
        spec=spec,
        source=source,
        param_space=param_space,
        metrics=metrics,
        iteration=iteration,
        history_tail=tail,
        notes=dict(notes),
    )


MeasureFn = Callable[[CircuitSource, int], MetricsBundle]


@dataclass
class PatchLoopStrategy(Strategy):
    """Simple patch-apply-simulate loop strategy."""

    name: str = "patch_loop"
    version: str = "0.1.0"

    signature_op: Any = None
    patch_apply_op: Any = None
    patch_guard_op: Any = None
    topology_guard_op: Any = None
    behavior_guard_op: Any = None
    guard_chain_op: Any = None
    formal_guard_op: Any = None
    deck_build_op: Any = None
    sim_run_op: Any = None
    metrics_op: Any = None
    llm_call_op: Any = None
    registry: MetricRegistry | None = None
    policy: Policy | None = None
    measure_fn: Optional[MeasureFn] = None
    max_patch_retries: int = 2

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
        if self.llm_call_op is None:
            self.llm_call_op = LLMCallOperator()
        if self.registry is None:
            self.registry = DEFAULT_REGISTRY

    def _measure_metrics(
        self,
        source: CircuitSource,
        metric_groups: Mapping[SimKind, list[str]],
        ctx: Any,
        iter_idx: int,
        attempt_idx: int = 0,
        recorder: RunRecorder | None = None,
        manifest: Any = None,
    ) -> tuple[MetricsBundle, dict[str, str], list[str], int]:
        """Measure metrics either via injected measure_fn or actual operators.

        Returns: metrics bundle, stage_map, warnings, sim_run_count
        """
        if self.measure_fn is not None:
            return self.measure_fn(source, iter_idx), {}, [], 0

        bundles: list[MetricsBundle] = []
        stage_map: dict[str, str] = {}
        warnings: list[str] = []
        sim_runs = 0

        for kind, names in metric_groups.items():
            plan = _sim_plan_for_kind(kind)
            deck_res = self.deck_build_op.run({"circuit_source": source, "sim_plan": plan, "sim_kind": kind}, ctx=None)
            record_operator_result(recorder, deck_res)
            deck = deck_res.outputs["deck"]
            stage_name = f"{kind.value}_i{iter_idx:03d}_a{attempt_idx:02d}"
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

        return _merge_metrics(bundles), stage_map, warnings, sim_runs

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:  # type: ignore[override]
        if self.policy is None:
            raise ValueError("PatchLoopStrategy requires a policy instance")

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

        # Step 1: signature + IR
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

        # Step 2: ParamSpace
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
            manifest.environment.setdefault("policy_name", getattr(self.policy, "name", "unknown"))
            manifest.environment.setdefault("policy_version", getattr(self.policy, "version", "unknown"))
            manifest.environment.setdefault("strategy_name", self.name)
            manifest.environment.setdefault("strategy_version", self.version)
            if hasattr(self.policy, "provider"):
                manifest.environment.setdefault("llm_provider", getattr(self.policy, "provider"))
            if hasattr(self.policy, "model"):
                manifest.environment.setdefault("llm_model", getattr(self.policy, "model"))
            if hasattr(self.policy, "temperature"):
                manifest.environment.setdefault("llm_temperature", getattr(self.policy, "temperature"))
            if hasattr(self.policy, "max_tokens"):
                max_tokens = getattr(self.policy, "max_tokens")
                if max_tokens is not None:
                    manifest.environment.setdefault("llm_max_tokens", max_tokens)
            if hasattr(self.policy, "seed"):
                seed = getattr(self.policy, "seed")
                if seed is not None:
                    manifest.environment.setdefault("llm_seed", seed)
            if hasattr(self.policy, "response_schema_name"):
                manifest.environment.setdefault("llm_response_schema", getattr(self.policy, "response_schema_name"))
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
            if recorder is not None:
                recorder.write_input("source.sp", source.text)
                recorder.write_input("spec.json", spec_payload)
                recorder.write_input("param_space.json", param_payload)
                recorder.write_input("cfg.json", cfg_payload)
                recorder.write_input("signature.txt", signature)

        max_iters = cfg.budget.max_iterations
        max_sim_runs = cfg.budget.max_sim_runs
        history_tail_k = cfg.notes.get("history_tail_k", 5)

        # Step 3: metric grouping by SimKind
        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = _group_metric_names_by_kind(self.registry, metric_names)

        # Baseline measure (iteration 0) with guard/retry handling.
        baseline_attempts: list[dict[str, Any]] = []
        baseline_guard: GuardReport | None = None
        baseline_errors: list[str] = []
        metrics0 = MetricsBundle()
        stage_map0: dict[str, str] = {}
        warnings0: list[str] = []
        sim_runs = 0
        sim_runs_ok = 0
        sim_runs_failed = 0
        baseline_success = False
        baseline_stop_reason: StopReason | None = None

        for attempt in range(self.max_patch_retries + 1):
            if max_sim_runs is not None and sim_runs >= max_sim_runs:
                baseline_stop_reason = StopReason.budget_exhausted
                break
            try:
                metrics0, stage_map0, warnings0, sim_runs_delta = self._measure_metrics(
                    source,
                    metric_groups,
                    ctx,
                    iter_idx=0,
                    attempt_idx=attempt,
                    recorder=recorder,
                    manifest=manifest,
                )
                sim_runs += sim_runs_delta
                sim_runs_ok += sim_runs_delta
            except (SimulationError, MetricError, ValidationError) as exc:
                sim_runs += 1
                sim_runs_failed += 1
                check = GuardCheck(
                    name="behavior_guard",
                    ok=False,
                    severity="hard",
                    reasons=(str(exc),),
                    data={"error_type": type(exc).__name__},
                )
                baseline_guard_res = self.guard_chain_op.run({"checks": [check]}, ctx=None)
                record_operator_result(recorder, baseline_guard_res)
                baseline_guard = baseline_guard_res.outputs["report"]
                baseline_errors = guard_failures(baseline_guard)
                baseline_attempts.append(attempt_record(attempt, None, baseline_guard))
                if attempt >= self.max_patch_retries:
                    break
                continue

            behavior_check_res = self.behavior_guard_op.run(
                {"metrics": metrics0, "spec": spec, "stage_map": stage_map0, "guard_cfg": guard_cfg},
                ctx=None,
            )
            record_operator_result(recorder, behavior_check_res)
            behavior_check = behavior_check_res.outputs["check"]
            baseline_guard_res = self.guard_chain_op.run({"checks": [behavior_check]}, ctx=None)
            record_operator_result(recorder, baseline_guard_res)
            baseline_guard = baseline_guard_res.outputs["report"]
            baseline_attempts.append(attempt_record(attempt, None, baseline_guard, stage_map0, warnings0))
            baseline_errors = guard_failures(baseline_guard)

            if not baseline_guard.ok:
                if attempt >= self.max_patch_retries:
                    break
                continue

            baseline_success = True
            break

        if not baseline_success:
            if baseline_stop_reason is None:
                baseline_stop_reason = StopReason.guard_failed
            history.append(
                {
                    "iteration": 0,
                    "patch": None,
                    "signature_before": signature,
                    "signature_after": signature,
                    "metrics": {},
                    "score": float("inf"),
                    "all_pass": False,
                    "improved": False,
                    "objectives": [],
                    "sim_stages": stage_map0,
                    "warnings": warnings0,
                    "errors": baseline_errors,
                    "guard": guard_report_to_dict(baseline_guard) if baseline_guard else None,
                    "attempts": baseline_attempts,
                }
            )
            record_history_entry(recorder, history[-1])
            recording_errors = finalize_run(
                recorder=recorder,
                manifest=manifest,
                best_source=source,
                best_metrics=MetricsBundle(),
                history=history,
                stop_reason=baseline_stop_reason,
                best_score=float("inf"),
                best_iter=None,
                sim_runs=sim_runs,
                sim_runs_ok=sim_runs_ok,
                sim_runs_failed=sim_runs_failed,
            )
            return RunResult(
                best_source=source,
                best_metrics=MetricsBundle(),
                history=history,
                stop_reason=baseline_stop_reason,
                notes={
                    "best_score": float("inf"),
                    "all_pass": False,
                    "recording_errors": recording_errors,
                },
            )

        eval0 = evaluate_objectives(spec, metrics0)
        best_source = source
        best_metrics = metrics0
        best_score = eval0["score"]
        best_all_pass = eval0["all_pass"]
        best_iter = 0
        no_improve = 0
        stop_reason: StopReason | None = StopReason.reached_target if best_all_pass else None

        history.append(
            {
                "iteration": 0,
                "patch": None,
                "signature_before": signature,
                "signature_after": signature,
                "metrics": {k: v.value for k, v in metrics0.values.items()},
                "score": best_score,
                "all_pass": best_all_pass,
                "improved": False,
                "objectives": eval0["per_objective"],
                "sim_stages": stage_map0,
                "warnings": warnings0,
                "errors": baseline_errors,
                "guard": guard_report_to_dict(baseline_guard) if baseline_guard else None,
                "attempts": baseline_attempts,
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

        cur_source = source
        cur_signature = signature
        cur_metrics = metrics0

        for i in range(1, max_iters + 1):
            if max_sim_runs is not None and sim_runs >= max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            attempts: list[dict[str, Any]] = []
            errors: list[str] = []
            patch: Patch | None = None
            guard_report: GuardReport | None = None
            signature_before = cur_signature
            new_source = cur_source
            new_signature = cur_signature
            new_circuit_ir = circuit_ir
            metrics_i: MetricsBundle | None = None
            stage_map_i: dict[str, str] = {}
            warnings_i: list[str] = []
            sim_runs_delta = 0
            last_guard_failures: list[str] = []
            last_guard_report: GuardReport | None = None

            cur_eval = evaluate_objectives(spec, cur_metrics)
            current_score = cur_eval["score"]
            param_values, param_value_errors = extract_param_values(
                circuit_ir,
                param_ids=[p.param_id for p in param_space.params],
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

            success = False
            for attempt in range(self.max_patch_retries + 1):
                if max_sim_runs is not None and sim_runs >= max_sim_runs:
                    stop_reason = StopReason.budget_exhausted
                    break
                obs_notes: dict[str, Any] = {
                    "last_score": best_score,
                    "best_score": best_score,
                    "current_score": current_score,
                    "param_values": dict(param_values),
                    "attempt": attempt,
                }
                if param_value_errors:
                    obs_notes["param_value_errors"] = list(param_value_errors)
                if last_guard_failures:
                    obs_notes["last_guard_failures"] = list(last_guard_failures)
                if last_guard_report is not None:
                    obs_notes["last_guard_report"] = guard_report_to_dict(last_guard_report)

                obs = _make_observation(
                    spec=spec,
                    source=cur_source,
                    param_space=param_space,
                    metrics=cur_metrics,
                    iteration=i,
                    history=history,
                    history_tail_k=history_tail_k,
                    notes=obs_notes,
                )

                if _is_llm_policy(self.policy):
                    patch = _propose_llm_patch(self.policy, self.llm_call_op, obs, ctx)
                else:
                    patch = self.policy.propose(obs, ctx)
                if patch.stop:
                    stop_reason = StopReason.policy_stop
                    attempts.append(attempt_record(attempt, patch, None))
                    break

                new_source = cur_source
                new_signature = cur_signature
                new_circuit_ir = circuit_ir

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
                    last_guard_failures = guard_failures(guard_report)
                    last_guard_report = guard_report
                    attempts.append(attempt_record(attempt, patch, guard_report))
                    if attempt >= self.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                    continue

                try:
                    apply_res = self.patch_apply_op.run(
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
                    guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                    record_operator_result(recorder, guard_chain_res)
                    guard_report = guard_chain_res.outputs["report"]
                    last_guard_failures = guard_failures(guard_report)
                    last_guard_report = guard_report
                    attempts.append(attempt_record(attempt, patch, guard_report))
                    if attempt >= self.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                    continue

                topo_check_res = self.topology_guard_op.run(
                    {"signature_before": signature_before, "signature_after": new_signature},
                    ctx=None,
                )
                record_operator_result(recorder, topo_check_res)
                topo_check = topo_check_res.outputs["check"]
                checks.append(topo_check)
                if not topo_check.ok:
                    guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                    record_operator_result(recorder, guard_chain_res)
                    guard_report = guard_chain_res.outputs["report"]
                    last_guard_failures = guard_failures(guard_report)
                    last_guard_report = guard_report
                    attempts.append(attempt_record(attempt, patch, guard_report))
                    if attempt >= self.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                    continue

                if self.formal_guard_op is not None:
                    formal_check_res = self.formal_guard_op.run(
                        {"source": new_source, "circuit_ir": new_circuit_ir, "spec": spec},
                        ctx=None,
                    )
                    record_operator_result(recorder, formal_check_res)
                    formal_check = formal_check_res.outputs["check"]
                    checks.append(formal_check)

                try:
                    metrics_i, stage_map_i, warnings_i, sim_runs_delta = self._measure_metrics(
                        new_source,
                        metric_groups,
                        ctx,
                        iter_idx=i,
                        attempt_idx=attempt,
                        recorder=recorder,
                        manifest=manifest,
                    )
                    sim_runs += sim_runs_delta
                    sim_runs_ok += sim_runs_delta
                except (SimulationError, MetricError) as exc:
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
                    last_guard_failures = guard_failures(guard_report)
                    last_guard_report = guard_report
                    attempts.append(attempt_record(attempt, patch, guard_report))
                    if attempt >= self.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                    continue

                behavior_check_res = self.behavior_guard_op.run(
                    {"metrics": metrics_i, "spec": spec, "stage_map": stage_map_i, "guard_cfg": guard_cfg},
                    ctx=None,
                )
                record_operator_result(recorder, behavior_check_res)
                behavior_check = behavior_check_res.outputs["check"]
                checks.append(behavior_check)

                guard_chain_res = self.guard_chain_op.run({"checks": checks}, ctx=None)
                record_operator_result(recorder, guard_chain_res)
                guard_report = guard_chain_res.outputs["report"]
                attempts.append(attempt_record(attempt, patch, guard_report, stage_map_i, warnings_i))

                if not guard_report.ok:
                    last_guard_failures = guard_failures(guard_report)
                    last_guard_report = guard_report
                    if attempt >= self.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                    continue

                errors = guard_failures(guard_report)
                success = True
                last_guard_report = guard_report
                break

            if stop_reason in {StopReason.policy_stop, StopReason.guard_failed, StopReason.budget_exhausted}:
                errors = last_guard_failures
                if stop_reason == StopReason.budget_exhausted and not errors:
                    errors = ["budget_exhausted"]
                history.append(
                    {
                        "iteration": i,
                        "patch": patch_to_dict(patch) if patch else None,
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {k: v.value for k, v in cur_metrics.values.items()},
                        "score": best_score,
                        "all_pass": best_all_pass,
                        "improved": False,
                        "sim_stages": {},
                        "warnings": [],
                        "errors": errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                    }
                )
                record_history_entry(recorder, history[-1])
                break

            if not success or metrics_i is None:
                stop_reason = StopReason.guard_failed
                errors = last_guard_failures
                history.append(
                    {
                        "iteration": i,
                        "patch": patch_to_dict(patch) if patch else None,
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {k: v.value for k, v in cur_metrics.values.items()},
                        "score": best_score,
                        "all_pass": best_all_pass,
                        "improved": False,
                        "sim_stages": {},
                        "warnings": [],
                        "errors": errors,
                        "guard": guard_report_to_dict(guard_report) if guard_report else None,
                        "attempts": attempts,
                    }
                )
                record_history_entry(recorder, history[-1])
                break

            # Measure after patch (successful guard chain)
            cur_source = new_source
            circuit_ir = new_circuit_ir
            cur_signature = new_signature

            eval_i = evaluate_objectives(spec, metrics_i)
            improved = eval_i["score"] < best_score - 1e-12

            if improved:
                best_score = eval_i["score"]
                best_metrics = metrics_i
                best_source = cur_source
                best_all_pass = eval_i["all_pass"]
                best_iter = i
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "iteration": i,
                    "patch": patch_to_dict(patch) if patch else None,
                    "signature_before": signature_before,
                    "signature_after": cur_signature,
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

            cur_metrics = metrics_i

            if eval_i["all_pass"]:
                best_all_pass = True
                if eval_i["score"] <= best_score:
                    best_score = eval_i["score"]
                    best_metrics = metrics_i
                    best_source = cur_source
                    best_iter = i
                stop_reason = StopReason.reached_target
                break
            if cfg.budget.no_improve_patience and no_improve >= cfg.budget.no_improve_patience:
                stop_reason = StopReason.no_improvement
                break

        if stop_reason is None:
            stop_reason = StopReason.max_iterations

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
