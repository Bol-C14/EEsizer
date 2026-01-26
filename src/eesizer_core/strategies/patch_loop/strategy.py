from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ...contracts import CircuitSource, CircuitSpec, MetricsBundle, Patch, RunResult, StrategyConfig
from ...contracts.enums import StopReason
from ...contracts.policy import Policy
from ...contracts.strategy import Strategy
from ...contracts.provenance import stable_hash_json, stable_hash_str
from ...metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ...operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ...operators.guards import (
    PatchGuardOperator,
    TopologyGuardOperator,
    BehaviorGuardOperator,
    GuardChainOperator,
)
from ...operators.llm import LLMCallOperator
from ...sim import DeckBuildOperator, NgspiceRunOperator
from ...domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ...domain.spice.patching import extract_param_values
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
from ..attempt_pipeline import AttemptOperators, run_attempt
from .attempt import propose_patch
from .evaluate import MeasureFn, evaluate_metrics, run_baseline
from .planning import group_metric_names_by_kind, make_observation, extract_sim_plan
from .state import LoopResult, PatchLoopConfig, PatchLoopState


@dataclass
class _PatchLoopPrepared:
    history: list[dict[str, Any]]
    recorder: RunRecorder | None
    manifest: Any
    cfg: StrategyConfig
    source: CircuitSource
    circuit_ir: Any
    signature: str
    param_space: Any
    guard_cfg: dict[str, Any]
    config: PatchLoopConfig
    metric_groups: Mapping[Any, list[str]]
    sim_plan: Any | None
    param_ids: list[str]


@dataclass
class _PatchLoopBaselineOutcome:
    result: RunResult | None
    state: PatchLoopState | None
    stop_reason: StopReason | None


@dataclass
class _PatchLoopIterationOutcome:
    stop_reason: StopReason
    state: PatchLoopState


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
    measure_fn: MeasureFn | None = None
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

    def _prepare(
        self,
        spec: CircuitSpec,
        source: CircuitSource,
        ctx: Any,
        cfg: StrategyConfig,
    ) -> _PatchLoopPrepared:
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

        config = PatchLoopConfig(
            max_iters=cfg.budget.max_iterations,
            max_sim_runs=cfg.budget.max_sim_runs,
            history_tail_k=cfg.notes.get("history_tail_k", 5),
            max_patch_retries=self.max_patch_retries,
            guard_cfg=guard_cfg,
            validation_opts={
                "wl_ratio_min": guard_cfg.get("wl_ratio_min"),
                "max_mul_factor": guard_cfg.get("max_mul_factor", 10.0),
            },
            apply_opts={
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
            },
        )
        config.apply_opts["validation_opts"] = config.validation_opts

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)
        param_ids = [p.param_id for p in param_space.params]
        sim_plan = extract_sim_plan(spec.notes) or extract_sim_plan(cfg.notes)

        return _PatchLoopPrepared(
            history=history,
            recorder=recorder,
            manifest=manifest,
            cfg=cfg,
            source=source,
            circuit_ir=circuit_ir,
            signature=signature,
            param_space=param_space,
            guard_cfg=guard_cfg,
            config=config,
            metric_groups=metric_groups,
            sim_plan=sim_plan,
            param_ids=param_ids,
        )

    def _baseline_phase(
        self,
        prepared: _PatchLoopPrepared,
        spec: CircuitSpec,
        ctx: Any,
    ) -> _PatchLoopBaselineOutcome:
        baseline = run_baseline(
            source=prepared.source,
            spec=spec,
            metric_groups=prepared.metric_groups,
            ctx=ctx,
            guard_cfg=prepared.guard_cfg,
            max_retries=prepared.config.max_patch_retries,
            max_sim_runs=prepared.config.max_sim_runs,
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
            return _PatchLoopBaselineOutcome(
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
                state=None,
                stop_reason=baseline.stop_reason,
            )

        eval0 = evaluate_metrics(spec, baseline.metrics)
        stop_reason: StopReason | None = StopReason.reached_target if eval0["all_pass"] else None

        state = PatchLoopState(
            current_source=prepared.source,
            current_signature=prepared.signature,
            circuit_ir=prepared.circuit_ir,
            current_metrics=baseline.metrics,
            best_source=prepared.source,
            best_metrics=baseline.metrics,
            best_score=eval0["score"],
            best_all_pass=eval0["all_pass"],
            best_iter=0,
            history=prepared.history,
            sim_runs=baseline.sim_runs,
            sim_runs_ok=baseline.sim_runs_ok,
            sim_runs_failed=baseline.sim_runs_failed,
        )

        prepared.history.append(
            {
                "iteration": 0,
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

        return _PatchLoopBaselineOutcome(result=None, state=state, stop_reason=stop_reason)

    def _run_iterations(
        self,
        prepared: _PatchLoopPrepared,
        spec: CircuitSpec,
        ctx: Any,
        state: PatchLoopState,
    ) -> _PatchLoopIterationOutcome:
        config = prepared.config
        history = prepared.history
        recorder = prepared.recorder
        attempt_ops = AttemptOperators(
            patch_guard_op=self.patch_guard_op,
            patch_apply_op=self.patch_apply_op,
            topology_guard_op=self.topology_guard_op,
            behavior_guard_op=self.behavior_guard_op,
            guard_chain_op=self.guard_chain_op,
            formal_guard_op=self.formal_guard_op,
            deck_build_op=self.deck_build_op,
            sim_run_op=self.sim_run_op,
            metrics_op=self.metrics_op,
        )

        stop_reason: StopReason | None = None

        for i in range(1, config.max_iters + 1):
            if config.max_sim_runs is not None and state.sim_runs >= config.max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            attempts: list[dict[str, Any]] = []
            errors: list[str] = []
            patch: Patch | None = None
            guard_report = None
            signature_before = state.current_signature
            new_source = state.current_source
            new_signature = state.current_signature
            new_circuit_ir = state.circuit_ir
            metrics_i = None
            stage_map_i: dict[str, str] = {}
            warnings_i: list[str] = []
            last_guard_failures: list[str] = []
            last_guard_report = None

            cur_eval = evaluate_metrics(spec, state.current_metrics)
            current_score = cur_eval["score"]
            param_values, param_value_errors = extract_param_values(state.circuit_ir, param_ids=prepared.param_ids)

            success = False
            for attempt in range(config.max_patch_retries + 1):
                if config.max_sim_runs is not None and state.sim_runs >= config.max_sim_runs:
                    stop_reason = StopReason.budget_exhausted
                    break
                obs_notes: dict[str, Any] = {
                    "last_score": state.best_score,
                    "best_score": state.best_score,
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

                obs = make_observation(
                    spec=spec,
                    source=state.current_source,
                    param_space=prepared.param_space,
                    metrics=state.current_metrics,
                    iteration=i,
                    history=history,
                    history_tail_k=config.history_tail_k,
                    notes=obs_notes,
                )

                patch = propose_patch(self.policy, self.llm_call_op, obs, ctx)
                if patch.stop:
                    stop_reason = StopReason.policy_stop
                    attempts.append(attempt_record(attempt, patch, None))
                    break

                attempt_result = run_attempt(
                    iter_idx=i,
                    attempt=attempt,
                    patch=patch,
                    cur_source=state.current_source,
                    cur_signature=state.current_signature,
                    circuit_ir=state.circuit_ir,
                    param_space=prepared.param_space,
                    spec=spec,
                    guard_cfg=config.guard_cfg,
                    apply_opts=config.apply_opts,
                    metric_groups=prepared.metric_groups,
                    ctx=ctx,
                    recorder=recorder,
                    manifest=prepared.manifest,
                    measure_fn=self.measure_fn,
                    ops=attempt_ops,
                    sim_plan=prepared.sim_plan,
                )
                state.sim_runs += attempt_result.sim_runs
                state.sim_runs_ok += attempt_result.sim_runs_ok
                state.sim_runs_failed += attempt_result.sim_runs_failed

                guard_report = attempt_result.guard_report
                new_source = attempt_result.new_source
                new_signature = attempt_result.new_signature
                new_circuit_ir = attempt_result.new_circuit_ir
                metrics_i = attempt_result.metrics
                stage_map_i = attempt_result.stage_map
                warnings_i = attempt_result.warnings

                attempts.append(attempt_record(attempt, patch, guard_report, stage_map_i, warnings_i))

                if not attempt_result.success:
                    last_guard_failures = attempt_result.guard_failures
                    last_guard_report = guard_report
                    if attempt >= config.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                    continue

                errors = attempt_result.guard_failures
                last_guard_report = guard_report
                success = True
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
                        "metrics": {k: v.value for k, v in state.current_metrics.values.items()},
                        "score": state.best_score,
                        "all_pass": state.best_all_pass,
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
                        "metrics": {k: v.value for k, v in state.current_metrics.values.items()},
                        "score": state.best_score,
                        "all_pass": state.best_all_pass,
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

            state.current_source = new_source
            state.circuit_ir = new_circuit_ir
            state.current_signature = new_signature

            eval_i = evaluate_metrics(spec, metrics_i)
            improved = eval_i["score"] < state.best_score - 1e-12

            if improved:
                state.best_score = eval_i["score"]
                state.best_metrics = metrics_i
                state.best_source = state.current_source
                state.best_all_pass = eval_i["all_pass"]
                state.best_iter = i
                state.no_improve = 0
            else:
                state.no_improve += 1

            history.append(
                {
                    "iteration": i,
                    "patch": patch_to_dict(patch) if patch else None,
                    "signature_before": signature_before,
                    "signature_after": state.current_signature,
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

            state.current_metrics = metrics_i

            if eval_i["all_pass"]:
                state.best_all_pass = True
                if eval_i["score"] <= state.best_score:
                    state.best_score = eval_i["score"]
                    state.best_metrics = metrics_i
                    state.best_source = state.current_source
                    state.best_iter = i
                stop_reason = StopReason.reached_target
                break
            if prepared.cfg.budget.no_improve_patience and state.no_improve >= prepared.cfg.budget.no_improve_patience:
                stop_reason = StopReason.no_improvement
                break

        if stop_reason is None:
            stop_reason = StopReason.max_iterations

        return _PatchLoopIterationOutcome(stop_reason=stop_reason, state=state)

    def _finalize(
        self,
        prepared: _PatchLoopPrepared,
        state: PatchLoopState,
        stop_reason: StopReason,
    ) -> RunResult:
        loop_result = LoopResult(
            best_source=state.best_source,
            best_metrics=state.best_metrics,
            history=prepared.history,
            stop_reason=stop_reason,
            best_score=state.best_score,
            best_all_pass=state.best_all_pass,
            best_iter=state.best_iter,
            sim_runs=state.sim_runs,
            sim_runs_ok=state.sim_runs_ok,
            sim_runs_failed=state.sim_runs_failed,
        )

        recording_errors = finalize_run(
            recorder=prepared.recorder,
            manifest=prepared.manifest,
            best_source=loop_result.best_source,
            best_metrics=loop_result.best_metrics,
            history=loop_result.history,
            stop_reason=loop_result.stop_reason,
            best_score=loop_result.best_score,
            best_iter=loop_result.best_iter,
            sim_runs=loop_result.sim_runs,
            sim_runs_ok=loop_result.sim_runs_ok,
            sim_runs_failed=loop_result.sim_runs_failed,
        )

        return RunResult(
            best_source=loop_result.best_source,
            best_metrics=loop_result.best_metrics,
            history=loop_result.history,
            stop_reason=loop_result.stop_reason,
            notes={
                "best_score": loop_result.best_score,
                "all_pass": loop_result.best_all_pass,
                "recording_errors": recording_errors,
            },
        )

    def run(self, spec: CircuitSpec, source: CircuitSource, ctx: Any, cfg: StrategyConfig) -> RunResult:  # type: ignore[override]
        if self.policy is None:
            raise ValueError("PatchLoopStrategy requires a policy instance")

        prepared = self._prepare(spec, source, ctx, cfg)
        baseline_out = self._baseline_phase(prepared, spec, ctx)
        if baseline_out.result is not None:
            return baseline_out.result

        assert baseline_out.state is not None

        if baseline_out.stop_reason is StopReason.reached_target:
            return self._finalize(prepared, baseline_out.state, baseline_out.stop_reason)

        iteration_out = self._run_iterations(prepared, spec, ctx, baseline_out.state)
        return self._finalize(prepared, iteration_out.state, iteration_out.stop_reason)
