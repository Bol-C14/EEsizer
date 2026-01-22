from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from .base import Agent, AgentContext


def _etype_counts(circuit_ir) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for elem in circuit_ir.elements.values():
        etype = str(elem.etype)
        counts[etype] = counts.get(etype, 0) + 1
    return dict(sorted(counts.items()))


def _node_counts(circuit_ir) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for elem in circuit_ir.elements.values():
        for n in elem.nodes:
            counts[str(n)] = counts.get(str(n), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@dataclass
class CircuitAnalysisAgent:
    """Deterministic circuit summary generator."""

    name: str = "circuit_analysis"
    version: str = "0.1.0"

    def run(self, ctx: AgentContext, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        cir = ctx.circuit_ir

        etype_counts = _etype_counts(cir)
        node_counts = _node_counts(cir)
        param_ids = sorted(cir.param_locs.keys())
        includes = list(cir.includes or ())

        summary = {
            "signature": ctx.signature,
            "num_lines": len(cir.lines),
            "num_elements": len(cir.elements),
            "etype_counts": etype_counts,
            "num_params": len(param_ids),
            "includes": includes,
            "warnings": list(cir.warnings or ()),
            "top_nodes": list(node_counts.keys())[:10],
        }

        return {
            "circuit_summary": summary,
            "param_candidates": param_ids,
            "node_counts": node_counts,
        }
