from __future__ import annotations

from typing import Any, Mapping

from .artifacts import CircuitSpec
from .strategy import StrategyConfig
from .provenance import stable_hash_json


def _objective_payload(obj: Any) -> dict[str, Any]:
    return {
        "metric": obj.metric,
        "target": obj.target,
        "tol": obj.tol,
        "weight": obj.weight,
        "sense": obj.sense,
    }


def _constraint_payload(constraint: Any) -> dict[str, Any]:
    return {"kind": constraint.kind, "data": dict(constraint.data)}


def spec_payload(spec: CircuitSpec) -> dict[str, Any]:
    objectives = sorted((_objective_payload(o) for o in spec.objectives), key=lambda d: str(d.get("metric", "")).lower())
    constraints = sorted((_constraint_payload(c) for c in spec.constraints), key=lambda d: str(d.get("kind", "")).lower())
    return {
        "objectives": objectives,
        "constraints": constraints,
        "observables": list(spec.observables),
        "notes": dict(spec.notes),
    }


def cfg_payload(cfg: StrategyConfig) -> dict[str, Any]:
    budget = cfg.budget
    return {
        "budget": {
            "max_iterations": budget.max_iterations,
            "max_sim_runs": budget.max_sim_runs,
            "timeout_s": budget.timeout_s,
            "no_improve_patience": budget.no_improve_patience,
        },
        "seed": cfg.seed,
        "notes": dict(cfg.notes),
    }


def hash_spec(spec: CircuitSpec) -> str:
    return "sha256:" + stable_hash_json(spec_payload(spec))


def hash_cfg(cfg: StrategyConfig) -> str:
    return "sha256:" + stable_hash_json(cfg_payload(cfg))


def hash_payload(payload: Any) -> str:
    return "sha256:" + stable_hash_json(payload)


def extract_sha256(hash_value: str | None) -> str | None:
    if not hash_value:
        return None
    text = str(hash_value)
    if text.startswith("sha256:"):
        return text.split("sha256:", 1)[1]
    return text

