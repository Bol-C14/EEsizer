from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from ..domain.spice.params import ParamInferenceRules, infer_param_space_from_ir
from .base import AgentContext


def _lower_elem_map(circuit_ir) -> Dict[str, Any]:
    return {name.lower(): elem for name, elem in circuit_ir.elements.items()}


def _param_key(param_id: str) -> str:
    parts = param_id.split(".")
    return parts[-1].lower() if parts else param_id.lower()


def _pick_passive_rc_params(param_ids: list[str], circuit_ir) -> list[str]:
    """Pick a small, safe set of parameters for passive RC-style circuits."""
    elem_map = _lower_elem_map(circuit_ir)
    r_candidates: list[str] = []
    c_candidates: list[str] = []
    for pid in param_ids:
        base = pid.split(".", 1)[0].lower()
        elem = elem_map.get(base)
        if elem is None:
            continue
        if str(elem.etype).upper() == "R" and pid.endswith(".value"):
            r_candidates.append(pid)
        if str(elem.etype).upper() == "C" and pid.endswith(".value"):
            c_candidates.append(pid)
    out: list[str] = []
    if r_candidates:
        out.append(sorted(r_candidates)[0])
    if c_candidates:
        out.append(sorted(c_candidates)[0])
    return out


def _detect_passive_rc(circuit_ir) -> bool:
    etypes = {str(elem.etype).upper() for elem in circuit_ir.elements.values()}
    # Very conservative: RC only if no MOS and no subckts.
    if "MOS" in etypes or "SUBCKT" in etypes:
        return False
    return "R" in etypes and "C" in etypes


def _regex_union_exact(names: list[str]) -> str:
    parts = [re.escape(n) for n in sorted(set(names))]
    if not parts:
        return r"$^"  # match nothing
    if len(parts) == 1:
        return rf"^({parts[0]})$"
    return rf"^({'|'.join(parts)})$"


@dataclass
class KnobAgent:
    """Produce a conservative ParamInferenceRules allowlist + ParamSpace."""

    name: str = "knobs"
    version: str = "0.1.0"
    max_params_default: int = 6

    def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        # If user already provided explicit param_rules, respect them.
        notes = ctx.cfg.notes or {}
        user_rules = notes.get("param_rules")
        if isinstance(user_rules, Mapping) and user_rules.get("allow_patterns"):
            rules = dict(user_rules)
            selected: list[str] = []
            try:
                ps = infer_param_space_from_ir(ctx.circuit_ir, rules=ParamInferenceRules(**rules))
            except Exception:
                ps = infer_param_space_from_ir(ctx.circuit_ir)
            return {"param_rules": rules, "param_space": ps, "selected_param_ids": selected, "mode": "user"}

        param_candidates = list(inputs.get("param_candidates") or sorted(ctx.circuit_ir.param_locs.keys()))
        max_params = int(notes.get("orchestrator", {}).get("max_params", self.max_params_default)) if isinstance(notes.get("orchestrator"), Mapping) else self.max_params_default

        selected: list[str] = []
        mode = "generic"
        if _detect_passive_rc(ctx.circuit_ir):
            selected = _pick_passive_rc_params(param_candidates, ctx.circuit_ir)
            mode = "passive_rc"

        if not selected:
            # Generic safe keys.
            safe_keys = {"value", "w", "l", "m", "nf"}
            for pid in param_candidates:
                if _param_key(pid) in safe_keys:
                    selected.append(pid)
                if len(selected) >= max_params:
                    break

        # Final fallback: take the first few params.
        if not selected:
            selected = param_candidates[:max_params]

        allow_pattern = _regex_union_exact(selected)
        rules = {"allow_patterns": [allow_pattern]}
        ps = infer_param_space_from_ir(ctx.circuit_ir, rules=ParamInferenceRules(**rules))

        return {
            "param_rules": rules,
            "param_space": ps,
            "selected_param_ids": selected,
            "mode": mode,
        }
