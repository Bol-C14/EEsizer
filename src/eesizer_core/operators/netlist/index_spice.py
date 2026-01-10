from __future__ import annotations

from typing import Any, Mapping

from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.artifacts import TokenLoc
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ...domain.spice import index_spice_netlist
from ...domain.spice.sanitize_rules import has_control_block


def _param_locs_payload(param_locs: dict[str, TokenLoc]) -> dict[str, object]:
    return {
        name: {
            "line_idx": loc.line_idx,
            "token_idx": loc.token_idx,
            "key": loc.key,
            "raw_token": loc.raw_token,
            "value_span": list(loc.value_span),
        }
        for name, loc in sorted(param_locs.items())
    }


class SpiceIndexOperator(Operator):
    """Operator that indexes a sanitized SPICE netlist into CircuitIR."""

    name = "index_spice_netlist"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        netlist_text = inputs.get("netlist_text")
        if not isinstance(netlist_text, str):
            raise ValidationError("netlist_text must be provided as a string")

        if has_control_block(netlist_text):
            raise ValidationError("netlist_text contains .control/.endc; sanitize first")

        includes = inputs.get("includes")
        if includes is not None and not isinstance(includes, (list, tuple)):
            raise ValidationError("includes must be a list/tuple of strings if provided")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["netlist_text"] = ArtifactFingerprint(sha256=stable_hash_str(netlist_text))
        if includes is not None:
            provenance.inputs["includes"] = ArtifactFingerprint(sha256=stable_hash_json(tuple(includes)))

        circuit_ir = index_spice_netlist(netlist_text, includes=tuple(includes) if includes is not None else None)
        provenance.outputs["circuit_ir"] = ArtifactFingerprint(
            sha256=stable_hash_json(_param_locs_payload(circuit_ir.param_locs))
        )
        provenance.finish()

        return OperatorResult(
            outputs={
                "circuit_ir": circuit_ir,
            },
            provenance=provenance,
            warnings=list(circuit_ir.warnings),
        )
