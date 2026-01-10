from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Tuple

from ...contracts.artifacts import CircuitIR, Element
from .parse import index_spice_netlist
from .sanitize_rules import SanitizeResult, sanitize_spice_netlist


@dataclass(frozen=True)
class TopologySignatureResult:
    signature: str
    circuit_ir: CircuitIR
    sanitize_result: SanitizeResult


def _normalize_node(node: str) -> str:
    return node.strip().lower()


def _element_fingerprint(elem: Element) -> Tuple[str, str, Tuple[str, ...], str, Tuple[str, ...]]:
    nodes_normalized = tuple(_normalize_node(n) for n in elem.nodes)
    param_keys = tuple(sorted(elem.params.keys()))
    return (
        elem.etype,
        elem.name,
        nodes_normalized,
        elem.model_or_subckt or "",
        param_keys,
    )


def _signature_payload(cir: CircuitIR, include_paths: bool) -> Dict[str, object]:
    elements_fp = [_element_fingerprint(elem) for _, elem in sorted(cir.elements.items())]
    payload: Dict[str, object] = {"elements": elements_fp}
    if include_paths:
        payload["includes"] = sorted(cir.includes)
    return payload


def topology_signature(netlist_text: str, include_paths: bool = True, max_lines: int = 5000) -> TopologySignatureResult:
    """Compute a stable topology signature for a SPICE netlist."""
    sanitize_result = sanitize_spice_netlist(netlist_text, max_lines=max_lines)
    cir = index_spice_netlist(sanitize_result.sanitized_text, includes=sanitize_result.includes)
    payload = _signature_payload(cir, include_paths=include_paths)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return TopologySignatureResult(signature=digest, circuit_ir=cir, sanitize_result=sanitize_result)
