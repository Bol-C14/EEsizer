from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math

from ...runtime.run_loader import RunLoader


_MAIN_METRICS = ("ugbw_hz", "phase_margin_deg", "power_w")


@dataclass(frozen=True)
class PlotContext:
    rows: list[dict[str, Any]]
    param_ids: list[str]
    nominal_values: dict[str, float]
    topk_iters: set[int]
    pareto_iters: set[int]
    best_iter: int | None
    spec_payload: dict[str, Any]
    robust_rows: list[dict[str, Any]]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_entry(entry: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    guard = entry.get("guard") or {}
    if isinstance(guard, Mapping) and guard:
        if guard.get("ok") is False:
            reasons: list[str] = []
            for check in guard.get("checks", []) or []:
                for reason in check.get("reasons", []) or []:
                    reasons.append(str(reason))
            if any("metric '" in reason or "metric \"" in reason for reason in reasons):
                return "metric_missing"
            if any("measurement_failed" in reason or "SimulationError" in reason for reason in reasons):
                return "sim_fail"
            return "guard_fail"

    missing = [name for name, value in metrics.items() if value is None]
    if missing:
        return "metric_missing"
    if entry.get("all_pass") is False:
        return "objective_fail"
    return "ok"


def _delta_payload(
    candidate: Mapping[str, Any],
    nominal_values: Mapping[str, float],
    param_ids: Iterable[str],
) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for param_id in param_ids:
        value = _safe_float(candidate.get(param_id))
        nominal = nominal_values.get(param_id)
        ratio = None
        delta_log = None
        delta = None
        if value is not None and nominal is not None:
            delta = value - nominal
            if nominal != 0:
                ratio = value / nominal
                if ratio is not None and ratio > 0:
                    delta_log = math.log10(ratio)
        deltas[param_id] = {
            "value": value,
            "nominal": nominal,
            "ratio": ratio,
            "log10": delta_log,
            "delta": delta,
        }
    return deltas


def _parse_spec_payload(run_dir: Path) -> dict[str, Any]:
    spec_path = run_dir / "inputs" / "spec.json"
    payload = _read_json(spec_path)
    return payload if isinstance(payload, dict) else {}


def _extract_param_ids(ranges: list[Mapping[str, Any]], candidates: list[Mapping[str, Any]]) -> list[str]:
    param_ids = [r.get("param_id") for r in ranges if not r.get("skipped")]
    param_ids = [str(pid) for pid in param_ids if pid]
    if param_ids:
        return param_ids
    ids: set[str] = set()
    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue
        for key in cand.keys():
            ids.add(str(key))
    return sorted(ids)


def _extract_nominals(ranges: list[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for entry in ranges:
        param_id = entry.get("param_id")
        if not param_id:
            continue
        nominal = _safe_float(entry.get("nominal"))
        if nominal is None:
            continue
        values[str(param_id)] = nominal
    return values


def _best_iter(history: Iterable[Mapping[str, Any]]) -> int | None:
    best_iter = None
    best_score = float("inf")
    for entry in history:
        iteration = entry.get("iteration")
        if iteration is None or iteration == 0:
            continue
        score = _safe_float(entry.get("score"))
        if score is None:
            continue
        if score < best_score:
            best_score = score
            best_iter = int(iteration)
    return best_iter


def _extract_robust_rows(history: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in history:
        corners = entry.get("corners")
        if not isinstance(corners, list) or not corners:
            continue
        nominal = None
        worst = None
        worst_score = float("-inf")
        for corner in corners:
            if not isinstance(corner, Mapping):
                continue
            if corner.get("corner_id") == "nominal":
                nominal = corner
            score = _safe_float(corner.get("score"))
            if score is None:
                continue
            if score > worst_score:
                worst_score = score
                worst = corner
        if nominal is None or worst is None:
            continue
        rows.append(
            {
                "iteration": entry.get("iteration"),
                "nominal_metrics": dict(nominal.get("metrics") or {}),
                "worst_metrics": dict(worst.get("metrics") or {}),
                "worst_corner_id": worst.get("corner_id"),
            }
        )
    return rows


def extract_plot_context(run_dir: Path) -> PlotContext:
    run_dir = Path(run_dir)
    loader = RunLoader(run_dir)
    history_entries = list(loader.iter_history())

    ranges_payload = _read_json(run_dir / "search" / "ranges.json")
    candidates_payload = _read_json(run_dir / "search" / "candidates.json")
    topk_payload = _read_json(run_dir / "search" / "topk.json")
    pareto_payload = _read_json(run_dir / "search" / "pareto.json")

    ranges = ranges_payload if isinstance(ranges_payload, list) else []
    candidates = candidates_payload if isinstance(candidates_payload, list) else []
    topk = topk_payload if isinstance(topk_payload, list) else []
    pareto = pareto_payload if isinstance(pareto_payload, list) else []

    param_ids = _extract_param_ids(ranges, candidates)
    nominal_values = _extract_nominals(ranges)
    topk_iters = {int(entry.get("iteration")) for entry in topk if isinstance(entry, Mapping) and entry.get("iteration") is not None}
    pareto_iters = {int(entry.get("iteration")) for entry in pareto if isinstance(entry, Mapping) and entry.get("iteration") is not None}
    best_iter = _best_iter(history_entries)

    rows: list[dict[str, Any]] = []
    for entry in history_entries:
        iteration = entry.get("iteration")
        if iteration is None:
            continue
        candidate = entry.get("candidate") or {}
        if not isinstance(candidate, Mapping):
            candidate = {}
        metrics = entry.get("metrics") or {}
        if not isinstance(metrics, Mapping):
            metrics = {}

        deltas = _delta_payload(candidate, nominal_values, param_ids)
        tags: list[str] = []
        if iteration == 0:
            tags.append("baseline")
        if iteration in topk_iters:
            tags.append("topk")
        if iteration in pareto_iters:
            tags.append("pareto")
        if best_iter is not None and iteration == best_iter:
            tags.append("best")

        missing_metrics = [name for name in _MAIN_METRICS if name in metrics and metrics.get(name) is None]
        status = _classify_entry(entry, metrics)

        rows.append(
            {
                "iteration": int(iteration),
                "candidate": {str(k): _safe_float(v) for k, v in candidate.items()},
                "metrics": {str(k): _safe_float(v) for k, v in metrics.items()},
                "score": _safe_float(entry.get("score")),
                "all_pass": bool(entry.get("all_pass")) if entry.get("all_pass") is not None else False,
                "deltas": deltas,
                "tags": sorted(tags),
                "status": status,
                "missing_metrics": sorted(set(missing_metrics)),
            }
        )

    rows.sort(key=lambda r: r.get("iteration", 0))
    spec_payload = _parse_spec_payload(run_dir)
    robust_rows = _extract_robust_rows(history_entries)

    return PlotContext(
        rows=rows,
        param_ids=param_ids,
        nominal_values=nominal_values,
        topk_iters=topk_iters,
        pareto_iters=pareto_iters,
        best_iter=best_iter,
        spec_payload=spec_payload,
        robust_rows=robust_rows,
    )
