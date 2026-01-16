from __future__ import annotations

from typing import Any, Mapping

from ...contracts.errors import ValidationError
from ...contracts.guards import GuardCheck
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json, stable_hash_str


class TopologyGuardOperator(Operator):
    """Post-apply guard to ensure topology signature does not change."""

    name = "topology_guard"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        before = inputs.get("signature_before")
        after = inputs.get("signature_after")
        if not isinstance(before, str):
            raise ValidationError("TopologyGuardOperator: 'signature_before' must be str")
        if not isinstance(after, str):
            raise ValidationError("TopologyGuardOperator: 'signature_after' must be str")

        ok = before == after
        reasons: tuple[str, ...] = ()
        if not ok:
            reasons = (f"signature mismatch: before={before[:12]} after={after[:12]}",)

        check = GuardCheck(
            name="topology_guard",
            ok=ok,
            severity="hard",
            reasons=reasons,
            data={"signature_before": before, "signature_after": after},
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["signature_before"] = ArtifactFingerprint(sha256=stable_hash_str(before))
        provenance.inputs["signature_after"] = ArtifactFingerprint(sha256=stable_hash_str(after))
        provenance.outputs["check"] = ArtifactFingerprint(
            sha256=stable_hash_json({"ok": check.ok, "reasons": list(check.reasons)})
        )
        provenance.finish()

        return OperatorResult(outputs={"check": check}, provenance=provenance)
