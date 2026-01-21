from __future__ import annotations

from typing import Any, Iterable


def objective_losses(eval_dict: dict[str, Any]) -> list[float]:
    losses: list[float] = []
    for obj in eval_dict.get("per_objective", []) or []:
        weight = obj.get("weight", 1.0)
        penalty = obj.get("penalty", 0.0)
        try:
            loss = float(penalty) * float(weight)
        except (TypeError, ValueError):
            loss = 0.0
        losses.append(loss)
    return losses


def pareto_front(loss_vectors: Iterable[list[float]]) -> list[int]:
    vectors = list(loss_vectors)
    if not vectors:
        return []
    front: list[int] = []
    for i, vec_i in enumerate(vectors):
        dominated = False
        for j, vec_j in enumerate(vectors):
            if i == j:
                continue
            if _dominates(vec_j, vec_i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _dominates(a: list[float], b: list[float]) -> bool:
    if not a or not b:
        return False
    if len(a) != len(b):
        return False
    better_or_equal = all(x <= y for x, y in zip(a, b))
    strictly_better = any(x < y for x, y in zip(a, b))
    return better_or_equal and strictly_better


def top_k(entries: Iterable[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    if k <= 0:
        return []
    return sorted(list(entries), key=lambda item: item.get("score", float("inf")))[:k]
