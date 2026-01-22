from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ..contracts import CircuitSpec, Objective
from .base import AgentContext


def _detect_passive_rc(circuit_ir) -> bool:
    etypes = {str(elem.etype).upper() for elem in circuit_ir.elements.values()}
    if "MOS" in etypes or "SUBCKT" in etypes:
        return False
    return "R" in etypes and "C" in etypes


def _maybe_read_default_spec(notes: Mapping[str, Any]) -> CircuitSpec | None:
    raw = notes.get("orchestrator_default_spec")
    if not isinstance(raw, Mapping):
        return None
    # Minimal: allow overriding the RC target.
    try:
        target = float(raw.get("ac_mag_db_at_1k_target", -20.0))
    except Exception:
        target = -20.0
    return CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=target, sense="ge"),))


@dataclass
class SpecSynthAgent:
    """Heuristic spec synthesis for simple demo circuits.

    This is intentionally conservative. For unknown circuits, callers should
    provide an explicit spec.
    """

    name: str = "spec_synth"
    version: str = "0.1.0"

    def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        # If user provided a spec, preserve it.
        if ctx.spec is not None and ctx.spec.objectives:
            return {"spec": ctx.spec, "mode": "user"}

        notes = ctx.cfg.notes or {}
        override = _maybe_read_default_spec(notes)
        if override is not None:
            return {"spec": override, "mode": "override"}

        if _detect_passive_rc(ctx.circuit_ir):
            # For the RC example, keep a simple, deterministic objective.
            spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))
            return {"spec": spec, "mode": "passive_rc"}

        # Fallback: create a spec with a mild default objective. This may not be
        # meaningful for all circuits but keeps the pipeline runnable.
        spec = CircuitSpec(objectives=(Objective(metric="ac_mag_db_at_1k", target=-20.0, sense="ge"),))
        return {"spec": spec, "mode": "fallback"}
