from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..contracts import CircuitSource, CircuitSpec, MetricsBundle, ParamSpace, Patch, StrategyConfig
from ..contracts.enums import StopReason
from ..contracts.guards import GuardCheck, GuardReport
from .recorder import RunRecorder


def patch_to_dict(patch: Patch) -> dict[str, Any]:
    return {
        "ops": [
            {"param": op.param, "op": getattr(op.op, "value", op.op), "value": op.value, "why": getattr(op, "why", "")}
            for op in patch.ops
        ],
        "stop": patch.stop,
        "notes": patch.notes,
    }


def _objective_to_dict(obj: Any) -> dict[str, Any]:
    return {
        "metric": obj.metric,
        "target": obj.target,
        "tol": obj.tol,
        "weight": obj.weight,
        "sense": obj.sense,
    }


def _constraint_to_dict(constraint: Any) -> dict[str, Any]:
    return {"kind": constraint.kind, "data": dict(constraint.data)}


def spec_to_dict(spec: CircuitSpec) -> dict[str, Any]:
    return {
        "objectives": [_objective_to_dict(obj) for obj in spec.objectives],
        "constraints": [_constraint_to_dict(c) for c in spec.constraints],
        "observables": list(spec.observables),
        "notes": dict(spec.notes),
    }


def param_space_to_dict(param_space: ParamSpace) -> dict[str, Any]:
    return {
        "params": [
            {
                "param_id": p.param_id,
                "unit": p.unit,
                "lower": p.lower,
                "upper": p.upper,
                "frozen": p.frozen,
                "tags": list(p.tags),
            }
            for p in param_space.params
        ]
    }


def strategy_cfg_to_dict(cfg: StrategyConfig, guard_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "budget": {
            "max_iterations": cfg.budget.max_iterations,
            "max_sim_runs": cfg.budget.max_sim_runs,
            "timeout_s": cfg.budget.timeout_s,
            "no_improve_patience": cfg.budget.no_improve_patience,
        },
        "seed": cfg.seed,
        "notes": dict(cfg.notes),
        "derived": {"guard_cfg": dict(guard_cfg)},
    }


def metrics_to_dict(metrics: MetricsBundle) -> dict[str, Any]:
    return {
        name: {
            "value": mv.value,
            "unit": mv.unit,
            "passed": mv.passed,
            "details": dict(mv.details),
        }
        for name, mv in metrics.values.items()
    }


def _provenance_to_dict(prov: Any) -> dict[str, Any]:
    return {
        "operator": prov.operator,
        "version": prov.version,
        "start_time": prov.start_time,
        "end_time": prov.end_time,
        "duration_s": prov.duration_s(),
        "command": prov.command,
        "inputs": {k: v.sha256 for k, v in prov.inputs.items()},
        "outputs": {k: v.sha256 for k, v in prov.outputs.items()},
        "notes": dict(prov.notes),
    }


def record_operator_result(recorder: RunRecorder | None, result: Any) -> None:
    if recorder is None:
        return
    try:
        payload = _provenance_to_dict(result.provenance)
    except Exception:
        return
    recorder.append_jsonl("provenance/operator_calls.jsonl", payload)


def _collect_search_files(recorder: RunRecorder) -> list[str]:
    search_dir = recorder.run_dir / "search"
    if not search_dir.exists():
        return []
    files: list[str] = []
    for path in search_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".json", ".md"}:
            continue
        files.append(recorder.relpath(path))
    return sorted(files)


def _relativize_stage_map(stage_map: Mapping[str, str], recorder: RunRecorder | None) -> dict[str, str]:
    if recorder is None:
        return dict(stage_map)
    return {str(k): recorder.relpath(v) for k, v in stage_map.items()}


def _prepare_history_entry(entry: dict[str, Any], recorder: RunRecorder | None) -> dict[str, Any]:
    payload = dict(entry)
    if "sim_stages" in payload and isinstance(payload["sim_stages"], Mapping):
        payload["sim_stages"] = _relativize_stage_map(payload["sim_stages"], recorder)
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        updated_attempts = []
        for attempt in attempts:
            attempt_payload = dict(attempt)
            if "sim_stages" in attempt_payload and isinstance(attempt_payload["sim_stages"], Mapping):
                attempt_payload["sim_stages"] = _relativize_stage_map(attempt_payload["sim_stages"], recorder)
            updated_attempts.append(attempt_payload)
        payload["attempts"] = updated_attempts
    return payload


def record_history_entry(recorder: RunRecorder | None, entry: dict[str, Any]) -> None:
    if recorder is None:
        return
    payload = _prepare_history_entry(entry, recorder)
    recorder.append_jsonl("history/iterations.jsonl", payload)


def _count_guard_fails(history: list[dict[str, Any]]) -> dict[str, int]:
    hard = 0
    soft = 0
    for entry in history:
        guard = entry.get("guard")
        if not isinstance(guard, dict):
            continue
        hard += len(guard.get("hard_fails", []) or [])
        soft += len(guard.get("soft_fails", []) or [])
    return {"hard": hard, "soft": soft}


def collect_llm_files(recorder: RunRecorder | None) -> list[str]:
    if recorder is None:
        return []
    llm_dir = recorder.run_dir / "llm"
    if not llm_dir.exists():
        return []
    llm_files: list[str] = []
    for path in llm_dir.rglob("*"):
        if path.is_file():
            llm_files.append(recorder.relpath(path))
    return sorted(llm_files)


def finalize_run(
    recorder: RunRecorder | None,
    manifest: Any,
    best_source: CircuitSource | None,
    best_metrics: MetricsBundle,
    history: list[dict[str, Any]],
    stop_reason: StopReason | None,
    best_score: float,
    best_iter: int | None,
    sim_runs: int,
    sim_runs_ok: int,
    sim_runs_failed: int,
    best_metrics_payload: Mapping[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    summary = {
        "stop_reason": stop_reason.value if stop_reason else None,
        "best_iter": best_iter,
        "best_score": best_score,
        "sim_runs_total": sim_runs,
        "sim_runs_ok": sim_runs_ok,
        "sim_runs_failed": sim_runs_failed,
        "guard_fail_counts": _count_guard_fails(history),
    }
    if recorder is not None:
        try:
            recorder.write_json("history/summary.json", summary)
        except Exception as exc:
            errors.append(f"summary_write_failed: {exc}")
        if best_source is not None:
            try:
                recorder.write_text("best/best.sp", best_source.text)
            except Exception as exc:
                errors.append(f"best_sp_write_failed: {exc}")
        try:
            payload = best_metrics_payload if best_metrics_payload is not None else metrics_to_dict(best_metrics)
            recorder.write_json("best/best_metrics.json", payload)
        except Exception as exc:
            errors.append(f"best_metrics_write_failed: {exc}")
        if errors:
            summary["recording_errors"] = list(errors)
            try:
                recorder.write_json("history/summary.json", summary)
            except Exception:
                pass
    if manifest is not None:
        manifest.result_summary = summary
        manifest.timestamp_end = datetime.now(timezone.utc).isoformat()
        if recorder is not None:
            llm_files = collect_llm_files(recorder)
            if llm_files:
                manifest.files["llm"] = llm_files
            search_files = _collect_search_files(recorder)
            for rel_path in search_files:
                manifest.files.setdefault(rel_path, rel_path)
        try:
            if recorder is not None:
                recorder.write_json("run_manifest.json", manifest.to_dict())
            else:
                manifest.save_json(Path(manifest.workspace) / "run_manifest.json")
        except Exception as exc:
            errors.append(f"manifest_write_failed: {exc}")
    return errors


def _guard_check_to_dict(check: GuardCheck) -> dict[str, Any]:
    return {
        "name": check.name,
        "ok": check.ok,
        "severity": check.severity,
        "reasons": list(check.reasons),
        "data": dict(check.data),
    }


def guard_report_to_dict(report: GuardReport) -> dict[str, Any]:
    return {
        "ok": report.ok,
        "checks": [_guard_check_to_dict(c) for c in report.checks],
        "hard_fails": [c.name for c in report.hard_fails],
        "soft_fails": [c.name for c in report.soft_fails],
    }


def guard_failures(report: GuardReport) -> list[str]:
    reasons: list[str] = []
    for check in report.hard_fails:
        reasons.extend(check.reasons)
    if not reasons:
        for check in report.soft_fails:
            reasons.extend(check.reasons)
    return reasons


def attempt_record(
    attempt: int,
    patch: Patch | None,
    report: GuardReport | None,
    stage_map: Mapping[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "patch": patch_to_dict(patch) if patch else None,
        "guard": guard_report_to_dict(report) if report else None,
        "errors": guard_failures(report) if report else [],
        "sim_stages": dict(stage_map or {}),
        "warnings": list(warnings or []),
    }
