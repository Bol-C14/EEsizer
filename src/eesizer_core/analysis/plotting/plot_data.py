from __future__ import annotations

from typing import Any, Iterable, Mapping


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_label(iteration: int, tags: Iterable[str]) -> str:
    tag_map = {"best": "B", "topk": "T", "pareto": "P", "baseline": "0"}
    suffix = "".join(tag_map[tag] for tag in ("best", "topk", "pareto") if tag in tags)
    if "baseline" in tags:
        suffix = "BASE"
    label = f"i{iteration}"
    if suffix:
        label = f"{label}:{suffix}"
    return label


def build_heatmap_data(rows: Iterable[Mapping[str, Any]], param_ids: list[str]) -> dict[str, Any]:
    heat_rows = []
    matrix: list[list[float | None]] = []
    row_labels: list[str] = []

    for row in rows:
        candidate = row.get("candidate") or {}
        if not candidate:
            continue
        iteration = int(row.get("iteration", 0))
        tags = row.get("tags") or []
        deltas = row.get("deltas") or {}
        values: list[float | None] = []
        for param_id in param_ids:
            delta = deltas.get(param_id, {}) if isinstance(deltas, Mapping) else {}
            delta_log = _safe_float(delta.get("log10"))
            values.append(delta_log)
        heat_rows.append(
            {
                "iteration": iteration,
                "tags": list(tags),
                "candidate": dict(candidate),
            }
        )
        matrix.append(values)
        row_labels.append(_row_label(iteration, tags))

    return {
        "plot": "knob_delta_heatmap",
        "value_kind": "log10_ratio",
        "param_ids": list(param_ids),
        "rows": heat_rows,
        "row_labels": row_labels,
        "matrix": matrix,
    }


def build_tradeoff_data(
    rows: Iterable[Mapping[str, Any]],
    *,
    x_metric: str,
    y_metric: str,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        x_val = _safe_float(metrics.get(x_metric))
        y_val = _safe_float(metrics.get(y_metric))
        points.append(
            {
                "iteration": int(row.get("iteration", 0)),
                "x": x_val,
                "y": y_val,
                "status": row.get("status", "ok"),
                "tags": list(row.get("tags") or []),
                "metrics": dict(metrics),
            }
        )
    return {
        "plot": f"tradeoff_{x_metric}_vs_{y_metric}",
        "x_metric": x_metric,
        "y_metric": y_metric,
        "points": points,
    }


def _pm_target(spec_payload: Mapping[str, Any]) -> float | None:
    objectives = spec_payload.get("objectives", []) if isinstance(spec_payload, Mapping) else []
    for obj in objectives or []:
        if not isinstance(obj, Mapping):
            continue
        metric = str(obj.get("metric", "")).strip().lower()
        if metric == "phase_margin_deg":
            return _safe_float(obj.get("target"))
    return None


def build_pm_vs_ugbw_data(rows: Iterable[Mapping[str, Any]], spec_payload: Mapping[str, Any]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        x_val = _safe_float(metrics.get("ugbw_hz"))
        y_val = _safe_float(metrics.get("phase_margin_deg"))
        points.append(
            {
                "iteration": int(row.get("iteration", 0)),
                "x": x_val,
                "y": y_val,
                "status": row.get("status", "ok"),
                "tags": list(row.get("tags") or []),
                "metrics": dict(metrics),
            }
        )
    return {
        "plot": "tradeoff_pm_vs_ugbw",
        "x_metric": "ugbw_hz",
        "y_metric": "phase_margin_deg",
        "pm_target": _pm_target(spec_payload),
        "points": points,
    }


def build_failure_breakdown_data(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "guard_fail": 0,
        "sim_fail": 0,
        "metric_missing": 0,
        "objective_fail": 0,
        "ok": 0,
    }
    for row in rows:
        if not row.get("candidate"):
            continue
        status = row.get("status", "ok")
        if status not in counts:
            status = "ok"
        counts[status] += 1
    return {
        "plot": "failures_breakdown",
        "counts": counts,
    }


def build_nominal_vs_worst_data(robust_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for row in robust_rows:
        nominal = row.get("nominal_metrics") or {}
        worst = row.get("worst_metrics") or {}
        points.append(
            {
                "iteration": row.get("iteration"),
                "worst_corner_id": row.get("worst_corner_id"),
                "nominal": {
                    "ugbw_hz": _safe_float(nominal.get("ugbw_hz")),
                    "power_w": _safe_float(nominal.get("power_w")),
                },
                "worst": {
                    "ugbw_hz": _safe_float(worst.get("ugbw_hz")),
                    "power_w": _safe_float(worst.get("power_w")),
                },
            }
        )
    return {
        "plot": "robust_nominal_vs_worst",
        "points": points,
    }
