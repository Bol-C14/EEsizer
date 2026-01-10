from __future__ import annotations

from typing import Any, Mapping

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ...domain.spice import index_spice_netlist, sanitize_spice_netlist, topology_signature


class SpiceCanonicalizeOperator(Operator):
    """Sanitize + index + signature in one shot to produce canonical artifacts."""

    name = "spice_canonicalize"
    version = "0.1.0"

    def __init__(self, include_paths: bool = True, max_lines: int = 5000) -> None:
        self.include_paths = include_paths
        self.max_lines = max_lines

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))

        sanitize_result = sanitize_spice_netlist(netlist_text, max_lines=self.max_lines)
        circuit_ir = index_spice_netlist(sanitize_result.sanitized_text, includes=sanitize_result.includes)
        sig_result = topology_signature(
            sanitize_result.sanitized_text,
            include_paths=self.include_paths,
            max_lines=self.max_lines,
        )

        provenance.outputs["sanitized_text"] = ArtifactFingerprint(sha256=stable_hash_str(sanitize_result.sanitized_text))
        provenance.outputs["includes"] = ArtifactFingerprint(sha256=stable_hash_json(sanitize_result.includes))
        provenance.outputs["circuit_ir"] = ArtifactFingerprint(
            sha256=stable_hash_json({k: v.value_span for k, v in circuit_ir.param_locs.items()})
        )
        provenance.outputs["signature"] = ArtifactFingerprint(sha256=stable_hash_str(sig_result.signature))
        provenance.finish()

        warnings = list(sanitize_result.warnings) + list(circuit_ir.warnings)

        return OperatorResult(
            outputs={
                "sanitized_text": sanitize_result.sanitized_text,
                "includes": sanitize_result.includes,
                "circuit_ir": circuit_ir,
                "topology_signature": sig_result.signature,
            },
            provenance=provenance,
            warnings=warnings,
        )
