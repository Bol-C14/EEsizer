from __future__ import annotations

from typing import Any, Mapping

from ...contracts.artifacts import CircuitIR, ParamSpace, Patch
from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_str
from ...domain.spice.patching import apply_patch_with_topology_guard, validate_patch


class PatchApplyOperator(Operator):
    """Apply a parameter Patch to a CircuitIR and return updated artifacts."""

    name = "patch_apply"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        circuit_ir = inputs.get("circuit_ir")
        if not isinstance(circuit_ir, CircuitIR):
            raise ValidationError("circuit_ir must be provided as a CircuitIR")
        param_space = inputs.get("param_space")
        if not isinstance(param_space, ParamSpace):
            raise ValidationError("param_space must be provided as a ParamSpace")
        patch = inputs.get("patch")
        if not isinstance(patch, Patch):
            raise ValidationError("patch must be provided as a Patch")

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["circuit_ir"] = ArtifactFingerprint(
            sha256=stable_hash_str(str(circuit_ir.param_locs))
        )
        provenance.inputs["param_space"] = ArtifactFingerprint(
            sha256=stable_hash_str(str(tuple(p.param_id for p in param_space.params)))
        )
        provenance.inputs["patch"] = patch.fingerprint()

        validation = validate_patch(circuit_ir, param_space, patch)
        if not validation.ok:
            raise ValidationError("; ".join(validation.errors))

        new_circuit_ir = apply_patch_with_topology_guard(circuit_ir, patch)
        netlist_text = "\n".join(new_circuit_ir.lines)

        provenance.outputs["circuit_ir"] = ArtifactFingerprint(
            sha256=stable_hash_str(str(new_circuit_ir.param_locs))
        )
        provenance.outputs["netlist_text"] = ArtifactFingerprint(
            sha256=stable_hash_str(netlist_text)
        )
        provenance.finish()

        return OperatorResult(
            outputs={
                "circuit_ir": new_circuit_ir,
                "netlist_text": netlist_text,
            },
            provenance=provenance,
        )
