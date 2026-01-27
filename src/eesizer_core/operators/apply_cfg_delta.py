from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts.deltas import CfgDelta
from ..contracts.errors import ValidationError
from ..contracts.hashes import hash_cfg
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json
from ..contracts.strategy import OptimizationBudget, StrategyConfig


def _deep_merge(base: Any, updates: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(updates, Mapping):
        out = dict(base)
        for k, v in updates.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return updates


def apply_cfg_delta(cfg: StrategyConfig, delta: CfgDelta) -> StrategyConfig:
    budget = cfg.budget
    budget_updates = delta.budget
    new_budget = OptimizationBudget(
        max_iterations=int(budget_updates.get("max_iterations", budget.max_iterations)),
        max_sim_runs=budget_updates.get("max_sim_runs", budget.max_sim_runs),
        timeout_s=budget_updates.get("timeout_s", budget.timeout_s),
        no_improve_patience=int(budget_updates.get("no_improve_patience", budget.no_improve_patience)),
    )
    new_notes = _deep_merge(cfg.notes, delta.notes)
    return StrategyConfig(
        budget=new_budget,
        seed=cfg.seed if delta.seed is None else int(delta.seed),
        notes=dict(new_notes) if isinstance(new_notes, Mapping) else dict(cfg.notes),
    )


@dataclass
class ApplyCfgDeltaOperator(Operator):
    name: str = "apply_cfg_delta"
    version: str = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        cfg = inputs.get("cfg")
        if not isinstance(cfg, StrategyConfig):
            raise ValidationError("ApplyCfgDeltaOperator requires 'cfg' StrategyConfig")

        delta_raw = inputs.get("cfg_delta") or inputs.get("delta") or {}
        if isinstance(delta_raw, CfgDelta):
            delta = delta_raw
        elif isinstance(delta_raw, Mapping):
            # Support the shorthand where keys at the top-level (except budget/seed/notes)
            # are treated as notes updates.
            raw = dict(delta_raw)
            notes_delta = dict(raw.get("notes") or {}) if isinstance(raw.get("notes"), Mapping) else {}
            for k, v in raw.items():
                if k in {"budget", "seed", "notes"}:
                    continue
                notes_delta[k] = v
            delta = CfgDelta.from_dict(
                {
                    "budget": raw.get("budget") or {},
                    "seed": raw.get("seed"),
                    "notes": notes_delta,
                }
            )
        else:
            raise ValidationError("ApplyCfgDeltaOperator requires cfg_delta dict")

        new_cfg = apply_cfg_delta(cfg, delta)

        prov = Provenance(operator=self.name, version=self.version)
        prov.inputs["cfg"] = ArtifactFingerprint(sha256=hash_cfg(cfg).split("sha256:", 1)[1])
        prov.inputs["cfg_delta"] = ArtifactFingerprint(sha256=stable_hash_json(delta.to_dict()))
        prov.outputs["new_cfg"] = ArtifactFingerprint(sha256=hash_cfg(new_cfg).split("sha256:", 1)[1])
        prov.finish()

        return OperatorResult(outputs={"cfg": new_cfg, "old_cfg": cfg, "cfg_delta": delta.to_dict()}, provenance=prov)

