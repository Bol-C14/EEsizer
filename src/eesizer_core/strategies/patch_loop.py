from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

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
from ..contracts.artifacts import PatchOp
from ..contracts.enums import SimKind, StopReason
from ..contracts.errors import ValidationError
from ..contracts.policy import Observation, Policy
from ..contracts.strategy import Strategy
from ..metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ..operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ..sim import DeckBuildOperator, NgspiceRunOperator
from ..strategies.objective_eval import evaluate_objectives
from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir


def _patch_to_dict(patch: Patch) -> dict[str, Any]:
    return {
        "ops": [
            {"param": op.param, "op": getattr(op.op, "value", op.op), "value": op.value, "why": getattr(op, "why", "")}
            for op in patch.ops
        ],
        "stop": patch.stop,
        "notes": patch.notes,
    }


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
    deck_build_op: Any = None
    sim_run_op: Any = None
    metrics_op: Any = None
    registry: MetricRegistry | None = None
    policy: Policy | None = None
    measure_fn: Optional[MeasureFn] = None
    max_patch_retries: int = 2

    def __post_init__(self) -> None:
        if self.signature_op is None:
            self.signature_op = TopologySignatureOperator()
        if self.patch_apply_op is None:
            self.patch_apply_op = PatchApplyOperator()
        if self.deck_build_op is None:
            self.deck_build_op = DeckBuildOperator()
        if self.sim_run_op is None:
            self.sim_run_op = NgspiceRunOperator()
        if self.metrics_op is None:
            self.metrics_op = ComputeMetricsOperator()
        if self.registry is None:
            self.registry = DEFAULT_REGISTRY

    def _measure_metrics(
        self,
        source: CircuitSource,
        metric_groups: Mapping[SimKind, list[str]],
        ctx: Any,
        iter_idx: int,
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
            deck = deck_res.outputs["deck"]
            stage_name = f"{kind.value}_i{iter_idx:03d}"
            run_res = self.sim_run_op.run({"deck": deck, "stage": stage_name}, ctx)
            raw = run_res.outputs["raw_data"]
            metrics_res = self.metrics_op.run({"raw_data": raw, "metric_names": names}, ctx=None)
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
        run_dir = ctx.run_dir() if hasattr(ctx, "run_dir") else None

        # Step 1: signature + IR
        sig_res = self.signature_op.run(
            {
                "netlist_text": source.text,
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
            },
            ctx=None,
        ).outputs
        circuit_ir = sig_res["circuit_ir"]
        signature = sig_res["signature"]

        # Step 2: ParamSpace
        rules = ParamInferenceRules(**cfg.notes.get("param_rules", {}))
        param_space = infer_param_space_from_ir(
            circuit_ir,
            rules=rules,
            frozen_param_ids=cfg.notes.get("frozen_param_ids", ()),
        )

        # Step 3: metric grouping by SimKind
        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = _group_metric_names_by_kind(self.registry, metric_names)

        # Baseline measure (iteration 0)
        metrics0, stage_map0, warnings0, sim_runs = self._measure_metrics(source, metric_groups, ctx, iter_idx=0)
        eval0 = evaluate_objectives(spec, metrics0)
        best_source = source
        best_metrics = metrics0
        best_score = eval0["score"]
        best_all_pass = eval0["all_pass"]
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
                "errors": [],
            }
        )

        if stop_reason is StopReason.reached_target:
            return RunResult(best_source=best_source, best_metrics=best_metrics, history=history, stop_reason=stop_reason)

        max_iters = cfg.budget.max_iterations
        max_sim_runs = cfg.budget.max_sim_runs
        history_tail_k = cfg.notes.get("history_tail_k", 5)

        cur_source = source
        cur_signature = signature
        cur_metrics = metrics0

        for i in range(1, max_iters + 1):
            if max_sim_runs is not None and sim_runs >= max_sim_runs:
                stop_reason = StopReason.budget_exhausted
                break

            obs = _make_observation(
                spec=spec,
                source=cur_source,
                param_space=param_space,
                metrics=cur_metrics,
                iteration=i,
                history=history,
                history_tail_k=history_tail_k,
                notes={"last_score": best_score},
            )

            # Policy propose with limited retries on validation errors.
            errors: list[str] = []
            patch: Patch | None = None
            signature_before = cur_signature
            new_source = cur_source
            new_signature = cur_signature
            new_circuit_ir = circuit_ir
            validation_opts = {
                "wl_ratio_min": rules.wl_ratio_min,
                "max_mul_factor": cfg.notes.get("max_mul_factor", 10.0),
            }
            apply_opts = {
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
                "validation_opts": validation_opts,
            }
            for attempt in range(self.max_patch_retries + 1):
                patch = self.policy.propose(obs, ctx)
                if patch.stop:
                    stop_reason = StopReason.policy_stop
                    break
                try:
                    apply_res = self.patch_apply_op.run(
                        {"source": cur_source, "param_space": param_space, "patch": patch, **apply_opts},
                        ctx=None,
                    ).outputs
                    new_source: CircuitSource = apply_res["source"]
                    new_signature = apply_res["topology_signature"]
                    new_circuit_ir = apply_res["circuit_ir"]
                    break
                except ValidationError as exc:
                    errors.append(str(exc))
                    if attempt >= self.max_patch_retries:
                        stop_reason = StopReason.guard_failed
                        patch = None
                        new_source = cur_source
                        new_signature = cur_signature
                        new_circuit_ir = circuit_ir
                        break
            else:
                stop_reason = StopReason.guard_failed
                patch = None
                new_source = cur_source
                new_signature = cur_signature
                new_circuit_ir = circuit_ir

            if stop_reason == StopReason.policy_stop or stop_reason == StopReason.guard_failed:
                history.append(
                    {
                        "iteration": i,
                        "patch": _patch_to_dict(patch) if patch else None,
                        "signature_before": signature_before,
                        "signature_after": new_signature,
                        "metrics": {k: v.value for k, v in cur_metrics.values.items()},
                        "score": best_score,
                        "all_pass": best_all_pass,
                        "improved": False,
                        "sim_stages": {},
                        "warnings": [],
                        "errors": errors,
                    }
                )
                break

            # Measure after patch
            cur_source = new_source
            circuit_ir = new_circuit_ir
            cur_signature = new_signature

            metrics_i, stage_map_i, warnings_i, sim_runs_delta = self._measure_metrics(
                cur_source, metric_groups, ctx, iter_idx=i
            )
            sim_runs += sim_runs_delta
            eval_i = evaluate_objectives(spec, metrics_i)
            improved = eval_i["score"] < best_score - 1e-12

            if improved:
                best_score = eval_i["score"]
                best_metrics = metrics_i
                best_source = cur_source
                best_all_pass = eval_i["all_pass"]
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "iteration": i,
                    "patch": _patch_to_dict(patch) if patch else None,
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
                }
            )

            cur_metrics = metrics_i

            if eval_i["all_pass"]:
                best_all_pass = True
                if eval_i["score"] <= best_score:
                    best_score = eval_i["score"]
                    best_metrics = metrics_i
                    best_source = cur_source
                stop_reason = StopReason.reached_target
                break
            if cfg.budget.no_improve_patience and no_improve >= cfg.budget.no_improve_patience:
                stop_reason = StopReason.no_improvement
                break

        if stop_reason is None:
            stop_reason = StopReason.max_iterations

        # Save manifest if available
        if hasattr(ctx, "manifest") and run_dir is not None:
            try:
                ctx.manifest().save_json(run_dir / "run_manifest.json")  # type: ignore[arg-type]
            except Exception:
                pass

        return RunResult(
            best_source=best_source,
            best_metrics=best_metrics,
            history=history,
            stop_reason=stop_reason,
            notes={"best_score": best_score, "all_pass": best_all_pass},
        )
