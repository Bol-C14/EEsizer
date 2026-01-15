from __future__ import annotations

from typing import Any, Mapping

from ...contracts.artifacts import CircuitSource, ParamSpace, Patch, TokenLoc
from ...contracts.errors import ValidationError
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str
from ...domain.spice.patching import apply_patch_with_topology_guard, validate_patch
from ...domain.spice.signature import topology_signature
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


class PatchApplyOperator(Operator):
    """从 CircuitSource + ParamSpace + Patch 得到新的 CircuitSource + CircuitIR."""

    name = "patch_apply"
    version = "0.1.0"

    def __init__(self, include_paths: bool = True, max_lines: int = 5000) -> None:
        self._include_paths = include_paths
        self._max_lines = max_lines

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        src = inputs.get("source")
        if not isinstance(src, CircuitSource):
            raise ValidationError("PatchApplyOperator: 'source' must be CircuitSource")
        param_space = inputs.get("param_space")
        if not isinstance(param_space, ParamSpace):
            raise ValidationError("PatchApplyOperator: 'param_space' must be ParamSpace")
        patch = inputs.get("patch")
        if not isinstance(patch, Patch):
            raise ValidationError("PatchApplyOperator: 'patch' must be Patch")

        include_paths = inputs.get("include_paths", self._include_paths)
        max_lines = inputs.get("max_lines", self._max_lines)
        validation_opts = inputs.get("validation_opts", {})

        if has_control_block(src.text):
            raise ValidationError("PatchApplyOperator: source netlist must not contain .control/.endc blocks")

        sig = topology_signature(src.text, include_paths=include_paths, max_lines=max_lines)
        cir = sig.circuit_ir

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["source"] = src.fingerprint()
        provenance.inputs["param_space"] = ArtifactFingerprint(
            sha256=stable_hash_json([p.param_id for p in param_space.params])
        )
        provenance.inputs["patch"] = patch.fingerprint()

        validation = validate_patch(
            cir,
            param_space,
            patch,
            wl_ratio_min=validation_opts.get("wl_ratio_min"),
            max_mul_factor=validation_opts.get("max_mul_factor", 10.0),
        )
        if not validation.ok:
            raise ValidationError("; ".join(validation.errors))

        patched_circuit_ir = apply_patch_with_topology_guard(
            cir,
            patch,
            include_paths=include_paths,
            max_lines=max_lines,
        )
        netlist_text = "\n".join(patched_circuit_ir.lines)
        refreshed = topology_signature(netlist_text, include_paths=self._include_paths)
        new_circuit_ir = refreshed.circuit_ir
        new_source = CircuitSource(
            kind=src.kind,
            text=netlist_text,
            name=src.name,
            metadata=dict(src.metadata),
        )

        provenance.outputs["circuit_ir"] = ArtifactFingerprint(
            sha256=stable_hash_json(_param_locs_payload(new_circuit_ir.param_locs))
        )
        provenance.outputs["netlist_text"] = ArtifactFingerprint(
            sha256=stable_hash_str(netlist_text)
        )
        provenance.outputs["source"] = new_source.fingerprint()
        provenance.finish()

        return OperatorResult(
            outputs={
                "source": new_source,
                "circuit_ir": new_circuit_ir,
                "topology_signature": refreshed.signature,
            },
            provenance=provenance,
        )
