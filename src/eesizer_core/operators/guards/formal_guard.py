from __future__ import annotations

from typing import Any, Mapping

from ...contracts.guards import GuardCheck
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json


class FormalGuardOperator(Operator):
    """Placeholder for formal verification guard (not implemented)."""

    name = "formal_guard"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        check = GuardCheck(
            name="formal_guard",
            ok=True,
            severity="soft",
            reasons=("not_implemented",),
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.outputs["check"] = ArtifactFingerprint(
            sha256=stable_hash_json({"ok": check.ok, "severity": check.severity, "reasons": list(check.reasons)})
        )
        provenance.finish()

        return OperatorResult(outputs={"check": check}, provenance=provenance)
