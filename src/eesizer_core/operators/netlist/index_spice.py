from __future__ import annotations

from typing import Any, Mapping

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_str
from ...domain.spice import index_spice_netlist


class SpiceIndexOperator(Operator):
    """Operator that indexes a sanitized SPICE netlist into CircuitIR."""

    name = "index_spice_netlist"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")

        includes = inputs.get("includes")
        if includes is not None and not isinstance(includes, (list, tuple)):
            raise ValidationError("includes must be a list/tuple of strings if provided")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))
        if includes is not None:
            provenance.inputs["includes"] = ArtifactFingerprint(sha256=stable_hash_str(str(tuple(includes))))

        circuit_ir = index_spice_netlist(netlist_text, includes=tuple(includes) if includes is not None else None)
        provenance.outputs["circuit_ir"] = ArtifactFingerprint(sha256=stable_hash_str(str(circuit_ir.param_locs)))
        provenance.finish()

        return OperatorResult(
            outputs={
                "circuit_ir": circuit_ir,
            },
            provenance=provenance,
            warnings=list(circuit_ir.warnings),
        )
