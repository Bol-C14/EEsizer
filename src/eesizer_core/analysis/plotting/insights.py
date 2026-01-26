from __future__ import annotations

from typing import Any, Iterable, Mapping
import math


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda kv: kv[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            idx = indexed[k][0]
            ranks[idx] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(ys) < 3:
        return None
    rx = _rank(xs)
    ry = _rank(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def build_sensitivity_insights(
    rows: Iterable[Mapping[str, Any]],
    param_ids: Iterable[str],
    metric_names: Iterable[str],
) -> dict[str, Any]:
    results: dict[str, list[dict[str, Any]]] = {}

    for metric in metric_names:
        scores: list[dict[str, Any]] = []
        for param_id in param_ids:
            xs: list[float] = []
            ys: list[float] = []
            for row in rows:
                candidate = row.get("candidate") or {}
                if not candidate:
                    continue
                deltas = row.get("deltas") or {}
                delta = deltas.get(param_id, {}) if isinstance(deltas, Mapping) else {}
                value = _safe_float(delta.get("log10"))
                if value is None:
                    value = _safe_float(delta.get("ratio"))
                if value is None:
                    continue
                if metric == "score":
                    metric_val = _safe_float(row.get("score"))
                else:
                    metrics = row.get("metrics") or {}
                    metric_val = _safe_float(metrics.get(metric))
                if metric_val is None or not math.isfinite(metric_val):
                    continue
                xs.append(value)
                ys.append(metric_val)
            corr = _spearman(xs, ys)
            if corr is None:
                continue
            scores.append({"param_id": param_id, "spearman": corr, "n": len(xs)})
        scores.sort(key=lambda item: abs(item["spearman"]), reverse=True)
        if scores:
            results[metric] = scores[:3]

    return {
        "method": "spearman",
        "value_basis": "log10_ratio",
        "top_params": results,
    }
