from __future__ import annotations

from typing import Any, Mapping

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_str
from ...domain.spice import TopologySignatureResult, topology_signature


class TopologySignatureOperator(Operator):
    """Operator that computes a topology signature and circuit IR."""

    name = "topology_signature"
    version = "0.1.0"

    def __init__(self, include_paths: bool = True, max_lines: int = 5000) -> None:
        self.include_paths = include_paths
        self.max_lines = max_lines

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")

        include_paths = inputs.get("include_paths", self.include_paths)
        max_lines = inputs.get("max_lines", self.max_lines)
        if not isinstance(include_paths, bool):
            raise ValidationError("include_paths must be a boolean")
        if not isinstance(max_lines, int) or max_lines <= 0:
            raise ValidationError("max_lines must be a positive integer")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))
        provenance.inputs["include_paths"] = ArtifactFingerprint(sha256=stable_hash_str(str(include_paths)))
        provenance.inputs["max_lines"] = ArtifactFingerprint(sha256=stable_hash_str(str(max_lines)))

        result: TopologySignatureResult = topology_signature(
            netlist_text,
            include_paths=include_paths,
            max_lines=max_lines,
        )

        provenance.outputs["signature"] = ArtifactFingerprint(sha256=stable_hash_str(result.signature))
        provenance.outputs["circuit_ir"] = ArtifactFingerprint(sha256=stable_hash_str(str(result.circuit_ir.param_locs)))
        provenance.outputs["includes"] = ArtifactFingerprint(sha256=stable_hash_str(str(result.sanitize_result.includes)))
        provenance.finish()

        return OperatorResult(
            outputs={
                "signature": result.signature,
                "circuit_ir": result.circuit_ir,
                "includes": result.sanitize_result.includes,
                "warnings": result.sanitize_result.warnings + result.circuit_ir.warnings,
            },
            provenance=provenance,
            warnings=list(result.sanitize_result.warnings) + list(result.circuit_ir.warnings),
        )
