from __future__ import annotations

from typing import Any, Mapping

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_str
from ...domain.spice import SanitizeResult, sanitize_spice_netlist


class SpiceSanitizeOperator(Operator):
    """Operator wrapper around SPICE netlist sanitization."""

    name = "sanitize_spice_netlist"
    version = "0.1.0"

    def __init__(self, max_lines: int = 5000) -> None:
        self.max_lines = max_lines

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")

        max_lines = inputs.get("max_lines", self.max_lines)
        if not isinstance(max_lines, int) or max_lines <= 0:
            raise ValidationError("max_lines must be a positive integer")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))
        provenance.inputs["max_lines"] = ArtifactFingerprint(sha256=stable_hash_str(str(max_lines)))

        result: SanitizeResult = sanitize_spice_netlist(netlist_text, max_lines=max_lines)

        provenance.outputs["sanitized_text"] = ArtifactFingerprint(sha256=stable_hash_str(result.sanitized_text))
        provenance.outputs["includes"] = ArtifactFingerprint(sha256=stable_hash_str(str(result.includes)))
        provenance.finish()

        return OperatorResult(
            outputs={
                "sanitized_text": result.sanitized_text,
                "includes": result.includes,
                "warnings": result.warnings,
            },
            provenance=provenance,
            warnings=list(result.warnings),
        )
