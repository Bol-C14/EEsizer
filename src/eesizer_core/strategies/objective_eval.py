from __future__ import annotations

from typing import Any

from ..contracts import CircuitSpec, MetricsBundle
from ..metrics.aliases import canonicalize_metric_name


def _safe_abs(val: float | None) -> float:
    if val is None:
        return 0.0
    return abs(val)


def evaluate_objectives(spec: CircuitSpec, metrics: MetricsBundle, eps: float = 1e-12) -> dict[str, Any]:
    """
    Evaluate objectives against a MetricsBundle.

    Returns a dict:
    {
        "all_pass": bool,
        "score": float,
        "per_objective": [
            {
                "metric": str,
                "sense": "ge|le|eq",
                "target": float|None,
                "value": float|None,
                "passed": bool|None,
                "penalty": float,
                "weight": float,
            }
        ]
    }
    """
    per_obj: list[dict[str, Any]] = []
    total_score = 0.0
    all_pass = True

    for obj in spec.objectives:
        mv = metrics.get(obj.metric)
        if mv is None:
            mv = metrics.get(canonicalize_metric_name(obj.metric))
        val = mv.value if mv is not None else None
        sense = (obj.sense or "ge").lower()
        target = obj.target
        tol = obj.tol
        weight = obj.weight if obj.weight is not None else 1.0

        passed: bool | None = None
        penalty: float

        if target is None:
            # no target specified; treat as pass-through
            penalty = 0.0
            passed = True
        elif val is None:
            penalty = 1e6
            passed = False
        elif sense == "ge":
            gap = target - val
            penalty = max(0.0, gap / (_safe_abs(target) + eps))
            passed = gap <= 0
        elif sense == "le":
            gap = val - target
            penalty = max(0.0, gap / (_safe_abs(target) + eps))
            passed = gap <= 0
        elif sense == "eq":
            gap = abs(val - target)
            if tol is None:
                penalty = gap / (_safe_abs(target) + eps)
                passed = gap <= eps
            else:
                penalty = max(0.0, gap - tol) / (_safe_abs(target) + eps)
                passed = gap <= tol
        else:
            penalty = 1e6
            passed = False

        total_score += weight * penalty
        if not passed:
            all_pass = False

        per_obj.append(
            {
                "metric": obj.metric,
                "sense": sense,
                "target": target,
                "value": val,
                "passed": passed,
                "penalty": penalty,
                "weight": weight,
            }
        )

    return {"all_pass": all_pass, "score": total_score, "per_objective": per_obj}
