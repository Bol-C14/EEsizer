from __future__ import annotations

from typing import Any, Iterable, Mapping


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def aggregate_corner_results(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(results)
    if not items:
        return {
            "pass_rate": 0.0,
            "worst_score": float("inf"),
            "robust_losses": [],
            "worst_corner_id": None,
        }

    pass_count = 0
    worst_score = float("-inf")
    worst_corner_id = None
    losses_list: list[list[float]] = []

    for entry in items:
        if entry.get("all_pass"):
            pass_count += 1
        score = _coerce_float(entry.get("score", float("inf")), float("inf"))
        if score > worst_score or worst_corner_id is None:
            worst_score = score
            worst_corner_id = entry.get("corner_id")
        losses = entry.get("losses")
        if isinstance(losses, list):
            losses_list.append([_coerce_float(val, 0.0) for val in losses])
        else:
            losses_list.append([])

    max_len = max((len(losses) for losses in losses_list), default=0)
    robust_losses: list[float] = []
    for idx in range(max_len):
        vals = [losses[idx] if idx < len(losses) else 0.0 for losses in losses_list]
        robust_losses.append(max(vals) if vals else 0.0)

    pass_rate = pass_count / len(items)
    return {
        "pass_rate": pass_rate,
        "worst_score": worst_score,
        "robust_losses": robust_losses,
        "worst_corner_id": worst_corner_id,
    }
