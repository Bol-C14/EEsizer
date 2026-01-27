from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...analysis.corners import aggregate_corner_results
from ...analysis.objective_eval import evaluate_objectives
from ...analysis.pareto import pareto_front, top_k
from ...contracts import CircuitSource, CircuitSpec, MetricsBundle, Patch
from ...contracts.enums import StopReason
from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ...domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from ...domain.spice.patching import extract_param_values
from ...metrics import ComputeMetricsOperator, MetricRegistry, DEFAULT_REGISTRY
from ...operators.guards import BehaviorGuardOperator, GuardChainOperator, PatchGuardOperator, TopologyGuardOperator
from ...operators.netlist import PatchApplyOperator, TopologySignatureOperator
from ...operators.report_plots import ReportPlotsOperator
from ...runtime.recorder import RunRecorder
from ...runtime.recording_utils import record_operator_result
from ...search.corners import build_corner_set
from ...sim import DeckBuildOperator, NgspiceRunOperator
from .corners import (
    build_candidate_patch,
    build_corner_patch,
    resolve_corner_overrides,
    stage_tag_for_corner,
)
from .measurement import AttemptOperators, MeasureFn, evaluate_metrics, run_attempt
from .reporting import build_robustness_section, replace_section
from .sim_plan import extract_sim_plan, group_metric_names_by_kind
from .utils import read_json, safe_float


def _metrics_values(metrics: MetricsBundle | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return {name: mv.value for name, mv in metrics.values.items()}


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


def _extract_corner_param_ids(run_dir: Path, fallback: list[str]) -> list[str]:
    payload = read_json(run_dir / "search" / "ranges.json")
    ids: list[str] = []
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("skipped"):
                continue
            pid = entry.get("param_id")
            if isinstance(pid, str) and pid.strip():
                ids.append(pid.strip().lower())
    return ids or fallback


def _pick_oat_topm(run_dir: Path, *, param_ids: list[str], top_m: int) -> list[str]:
    if top_m <= 0:
        return []
    payload = read_json(run_dir / "insights" / "sensitivity.json")
    if isinstance(payload, Mapping):
        top_params = payload.get("top_params")
        if isinstance(top_params, Mapping):
            # Prefer score sensitivity if present.
            candidates = top_params.get("score") or next(iter(top_params.values()), [])
            picked: list[str] = []
            if isinstance(candidates, list):
                for item in candidates:
                    if not isinstance(item, Mapping):
                        continue
                    pid = item.get("param_id")
                    if not isinstance(pid, str):
                        continue
                    pid_norm = pid.lower()
                    if pid_norm in param_ids and pid_norm not in picked:
                        picked.append(pid_norm)
                    if len(picked) >= top_m:
                        break
            if picked:
                return picked
    return sorted(param_ids)[:top_m]


@dataclass(frozen=True)
class _CornerValidateConfig:
    candidates_source: str
    corners: str
    span_mul: float
    override_mode: str
    clamp_corner_overrides: bool
    top_m: int

    @staticmethod
    def from_inputs(inputs: Mapping[str, Any], cfg: Any) -> "_CornerValidateConfig":
        notes = cfg.notes.get("corner_validate") if hasattr(cfg, "notes") else None
        base = dict(notes) if isinstance(notes, Mapping) else {}
        override = inputs.get("corner_validate")
        if isinstance(override, Mapping):
            base.update(dict(override))

        candidates_source = str(inputs.get("candidates_source") or base.get("candidates_source") or "topk").lower()
        corners = str(base.get("corners", "oat")).lower()
        span_mul = float(base.get("span_mul", 10.0))
        override_mode = str(base.get("override_mode", "add")).lower()
        clamp = bool(base.get("clamp_corner_overrides", True))
        top_m = int(base.get("top_m", 3))
        return _CornerValidateConfig(
            candidates_source=candidates_source,
            corners=corners,
            span_mul=span_mul,
            override_mode=override_mode,
            clamp_corner_overrides=clamp,
            top_m=top_m,
        )


class CornerValidateOperator(Operator):
    """Validate grid candidates under deterministic corner sets and write robust artifacts."""

    name = "corner_validate"
    version = "0.1.0"

    def __init__(self, *, measure_fn: MeasureFn | None = None, registry: MetricRegistry | None = None) -> None:
        self.measure_fn = measure_fn
        self.registry = registry or DEFAULT_REGISTRY

        self.signature_op = TopologySignatureOperator()
        self.patch_apply_op = PatchApplyOperator()
        self.patch_guard_op = PatchGuardOperator()
        self.topology_guard_op = TopologyGuardOperator()
        self.behavior_guard_op = BehaviorGuardOperator()
        self.guard_chain_op = GuardChainOperator()
        self.deck_build_op = DeckBuildOperator()
        self.sim_run_op = NgspiceRunOperator()
        self.metrics_op = ComputeMetricsOperator(registry=self.registry)
        self.report_plots_op = ReportPlotsOperator()

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        run_dir = inputs.get("run_dir")
        recorder = inputs.get("recorder")
        manifest = inputs.get("manifest")
        source = inputs.get("source")
        spec = inputs.get("spec")
        cfg = inputs.get("cfg")

        if not isinstance(source, CircuitSource):
            raise ValidationError("CornerValidateOperator requires 'source' CircuitSource")
        if not isinstance(spec, CircuitSpec):
            raise ValidationError("CornerValidateOperator requires 'spec' CircuitSpec")
        if cfg is None or not hasattr(cfg, "notes") or not hasattr(cfg, "budget"):
            raise ValidationError("CornerValidateOperator requires 'cfg' StrategyConfig-like object")

        if recorder is None:
            if run_dir is None:
                raise ValidationError("CornerValidateOperator requires run_dir or recorder")
            recorder = RunRecorder(Path(run_dir))
        run_dir = recorder.run_dir

        cv_cfg = _CornerValidateConfig.from_inputs(inputs, cfg)

        report_file = run_dir / "report.md"
        report_text = report_file.read_text(encoding="utf-8") if report_file.exists() else ""

        # Prepare IR + signature (also triggers sanitize rules for includes).
        sig_result = self.signature_op.run(
            {
                "netlist_text": source.text,
                "include_paths": cfg.notes.get("include_paths", True),
                "max_lines": cfg.notes.get("max_lines", 50000),
            },
            ctx=None,
        )
        record_operator_result(recorder, sig_result)
        sig_out = sig_result.outputs
        circuit_ir = sig_out["circuit_ir"]
        signature = sig_out["signature"]
        signature_result = sig_out.get("signature_result")
        if signature_result is not None:
            sanitized_text = signature_result.sanitize_result.sanitized_text
            if sanitized_text != source.text:
                source = CircuitSource(
                    kind=source.kind,
                    text=sanitized_text,
                    name=source.name,
                    metadata=dict(source.metadata),
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

        validation_opts = {
            "wl_ratio_min": guard_cfg.get("wl_ratio_min"),
            "max_mul_factor": guard_cfg.get("max_mul_factor", 10.0),
        }
        apply_opts = {
            "include_paths": cfg.notes.get("include_paths", True),
            "max_lines": cfg.notes.get("max_lines", 50000),
            "validation_opts": validation_opts,
        }

        metric_names = [obj.metric for obj in spec.objectives]
        metric_groups = group_metric_names_by_kind(self.registry, metric_names)
        sim_plan = extract_sim_plan(spec.notes) or extract_sim_plan(cfg.notes)

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

        # Candidate list from grid artifacts.
        candidates_path = run_dir / "search" / f"{cv_cfg.candidates_source}.json"
        candidates_payload = read_json(candidates_path)
        candidate_entries = candidates_payload if isinstance(candidates_payload, list) else []

        # Corner parameter ids: prefer grid ranges when present.
        fallback_corner_param_ids = [p.param_id.lower() for p in param_space.params]
        corner_param_ids = _extract_corner_param_ids(run_dir, fallback_corner_param_ids)
        if cv_cfg.corners == "oat_topm":
            corner_param_ids = _pick_oat_topm(run_dir, param_ids=corner_param_ids, top_m=cv_cfg.top_m)

        # Corner set for validation (OAT builder already supports all_low/all_high).
        baseline_corner_values, _ = extract_param_values(circuit_ir, param_ids=corner_param_ids)
        corner_set = build_corner_set(
            param_space=param_space,
            nominal_values=baseline_corner_values,
            span_mul=cv_cfg.span_mul,
            corner_param_ids=corner_param_ids,
            include_global_corners=True,
            override_mode=cv_cfg.override_mode,
            mode="oat",
        )
        corner_defs = corner_set.get("corners") or []
        if cv_cfg.corners == "global":
            corner_defs = [c for c in corner_defs if isinstance(c, Mapping) and c.get("corner_id") in {"nominal", "all_low", "all_high"}]
            corner_set = dict(corner_set)
            corner_set["corners"] = list(corner_defs)

        recorder.write_json("search/corner_set.json", corner_set)
        if manifest is not None:
            manifest.files.setdefault("search/corner_set.json", "search/corner_set.json")

        param_bounds = corner_set.get("param_bounds") if isinstance(corner_set, Mapping) else {}
        if not isinstance(param_bounds, Mapping):
            param_bounds = {}

        robust_candidates: list[dict[str, Any]] = []
        sim_runs = 0
        sim_runs_ok = 0
        sim_runs_failed = 0

        for entry in candidate_entries:
            if not isinstance(entry, Mapping):
                continue
            iter_idx = entry.get("iteration")
            if iter_idx is None:
                continue
            iteration = int(iter_idx)
            candidate = entry.get("candidate") or {}
            if not isinstance(candidate, Mapping):
                candidate = {}

            candidate_patch = build_candidate_patch(candidate)
            if not candidate_patch.ops:
                continue

            # Measure candidate nominal (avoid reusing grid outputs to keep this stage self-contained).
            cand_attempt = run_attempt(
                iter_idx=iteration,
                attempt=0,
                patch=candidate_patch,
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
                sim_plan=sim_plan,
                stage_tag=stage_tag_for_corner("nominal"),
            )
            sim_runs += cand_attempt.sim_runs
            sim_runs_ok += cand_attempt.sim_runs_ok
            sim_runs_failed += cand_attempt.sim_runs_failed

            corners: list[dict[str, Any]] = []
            eval0 = evaluate_metrics(spec, cand_attempt.metrics) if cand_attempt.metrics is not None else None
            corners.append(
                {
                    "corner_id": "nominal",
                    "overrides": {},
                    "metrics": _metrics_values(cand_attempt.metrics),
                    "score": eval0.get("score") if eval0 else float("inf"),
                    "all_pass": bool(eval0.get("all_pass")) if eval0 else False,
                    "losses": eval0.get("per_objective") and [] or [],
                    "objectives": eval0.get("per_objective", []) if eval0 else [],
                }
            )

            candidate_base_values, _ = extract_param_values(cand_attempt.new_circuit_ir, param_ids=corner_param_ids)

            for corner_def in corner_defs:
                if not isinstance(corner_def, Mapping):
                    continue
                corner_id = str(corner_def.get("corner_id", "corner"))
                if corner_id == "nominal":
                    continue
                overrides = corner_def.get("overrides")
                if not isinstance(overrides, Mapping):
                    overrides = {}

                applied, override_errors, override_warnings = resolve_corner_overrides(
                    base_values=candidate_base_values,
                    param_bounds=param_bounds,
                    overrides=overrides,
                    clamp=cv_cfg.clamp_corner_overrides,
                )
                if override_errors:
                    corners.append(
                        {
                            "corner_id": corner_id,
                            "overrides": dict(applied),
                            "metrics": {},
                            "score": float("inf"),
                            "all_pass": False,
                            "warnings": override_warnings,
                            "errors": override_errors,
                        }
                    )
                    continue

                corner_patch = build_corner_patch(applied)
                corner_attempt = run_attempt(
                    iter_idx=iteration,
                    attempt=0,
                    patch=corner_patch,
                    cur_source=cand_attempt.new_source,
                    cur_signature=cand_attempt.new_signature,
                    circuit_ir=cand_attempt.new_circuit_ir,
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
                    sim_plan=sim_plan,
                    stage_tag=stage_tag_for_corner(corner_id),
                )
                sim_runs += corner_attempt.sim_runs
                sim_runs_ok += corner_attempt.sim_runs_ok
                sim_runs_failed += corner_attempt.sim_runs_failed

                eval_i = (
                    evaluate_metrics(spec, corner_attempt.metrics)
                    if corner_attempt.metrics is not None and corner_attempt.success
                    else None
                )
                corners.append(
                    {
                        "corner_id": corner_id,
                        "overrides": dict(applied),
                        "metrics": _metrics_values(corner_attempt.metrics),
                        "score": eval_i.get("score") if eval_i else float("inf"),
                        "all_pass": bool(eval_i.get("all_pass")) if eval_i else False,
                        "objectives": eval_i.get("per_objective", []) if eval_i else [],
                        "warnings": override_warnings + corner_attempt.warnings,
                        "errors": override_errors + corner_attempt.guard_failures,
                    }
                )

            summary = aggregate_corner_results(corners)
            worst_entry = _pick_corner(corners, corner_id=summary.get("worst_corner_id"), worst=True)
            best_entry = _pick_corner(corners, worst=False)
            nominal_entry = _pick_corner(corners, corner_id="nominal", worst=False)

            robust_candidates.append(
                {
                    "iteration": iteration,
                    "candidate": {str(k): safe_float(v) for k, v in candidate.items()},
                    "patch": {"ops": [{"param": op.param, "op": op.op.value, "value": op.value} for op in candidate_patch.ops]},
                    "pass_rate": summary.get("pass_rate"),
                    "worst_score": summary.get("worst_score"),
                    "robust_losses": summary.get("robust_losses"),
                    "worst_corner_id": summary.get("worst_corner_id"),
                    "nominal_metrics": dict((nominal_entry or {}).get("metrics") or {}),
                    "worst_metrics": dict((worst_entry or {}).get("metrics") or {}),
                    "best_metrics": dict((best_entry or {}).get("metrics") or {}),
                }
            )

        robust_topk = top_k(robust_candidates, k=int(cfg.notes.get("corner_validate", {}).get("top_k", 5)))
        loss_vectors = [entry.get("robust_losses") or [] for entry in robust_candidates]
        pareto_idx = pareto_front([list(vec) if isinstance(vec, list) else [] for vec in loss_vectors])
        robust_pareto = [robust_candidates[i] for i in pareto_idx] if pareto_idx else []

        meta = {
            "candidates_source": cv_cfg.candidates_source,
            "corners": cv_cfg.corners,
            "override_mode": cv_cfg.override_mode,
            "clamp_corner_overrides": cv_cfg.clamp_corner_overrides,
            "corner_param_ids": list(corner_param_ids),
            "corner_count": len(corner_defs),
            "validated_candidates": len(robust_candidates),
            "sim_runs": sim_runs,
            "sim_runs_ok": sim_runs_ok,
            "sim_runs_failed": sim_runs_failed,
        }

        recorder.write_json("search/robust_candidates.json", robust_candidates)
        recorder.write_json("search/robust_topk.json", robust_topk)
        recorder.write_json("search/robust_pareto.json", robust_pareto)
        recorder.write_json("search/robust_meta.json", meta)

        if manifest is not None:
            for rel_path in (
                "search/robust_candidates.json",
                "search/robust_topk.json",
                "search/robust_pareto.json",
                "search/robust_meta.json",
            ):
                manifest.files.setdefault(rel_path, rel_path)

        # Update report with robustness section, then re-run plotting to render robust plots.
        robust_section = build_robustness_section(
            config={
                "candidates_source": cv_cfg.candidates_source,
                "corners": cv_cfg.corners,
                "override_mode": cv_cfg.override_mode,
                "clamp_corner_overrides": cv_cfg.clamp_corner_overrides,
                "top_m": cv_cfg.top_m,
            },
            corner_param_ids=corner_param_ids,
            corner_count=len(corner_defs),
            robust_topk=robust_topk,
        )
        report_text = replace_section(report_text, "## Robustness Validation", robust_section)
        recorder.write_text("report.md", report_text)

        try:
            plots_res = self.report_plots_op.run(
                {"run_dir": run_dir, "recorder": recorder, "manifest": manifest, "report_path": "report.md"},
                ctx=None,
            )
            record_operator_result(recorder, plots_res)
        except Exception:
            pass

        if manifest is not None:
            recorder.write_json("run_manifest.json", manifest.to_dict())

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["run_dir"] = ArtifactFingerprint(sha256=stable_hash_str(str(run_dir)))
        prov.inputs["source"] = source.fingerprint()
        prov.inputs["spec"] = ArtifactFingerprint(sha256=stable_hash_json({"objectives": [o.metric for o in spec.objectives]}))
        prov.outputs["robust_topk"] = ArtifactFingerprint(sha256=stable_hash_json(robust_topk))
        prov.finish()

        return OperatorResult(
            outputs={
                "robust_candidates": robust_candidates,
                "robust_topk": robust_topk,
                "robust_pareto": robust_pareto,
                "robust_meta": meta,
            },
            provenance=prov,
        )

