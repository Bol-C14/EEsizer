from __future__ import annotations

from typing import Any, Iterable, Mapping

from ...contracts.errors import ValidationError
from ...contracts.guards import GuardCheck, GuardReport
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json


class GuardChainOperator(Operator):
    """Aggregate multiple GuardChecks into a GuardReport."""

    name = "guard_chain"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        checks = inputs.get("checks", ())
        if isinstance(checks, GuardCheck):
            checks_iter: Iterable[GuardCheck] = (checks,)
        elif isinstance(checks, Iterable):
            checks_iter = checks  # type: ignore[assignment]
        else:
            raise ValidationError("GuardChainOperator: 'checks' must be iterable of GuardCheck")

        check_list: list[GuardCheck] = []
        for check in checks_iter:
            if not isinstance(check, GuardCheck):
                raise ValidationError("GuardChainOperator: all items in 'checks' must be GuardCheck")
            check_list.append(check)

        report = GuardReport(checks=tuple(check_list))

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["checks"] = ArtifactFingerprint(
            sha256=stable_hash_json([{"name": c.name, "ok": c.ok, "severity": c.severity} for c in check_list])
        )
        provenance.outputs["report"] = ArtifactFingerprint(
            sha256=stable_hash_json(
                {
                    "ok": report.ok,
                    "hard_fails": [c.name for c in report.hard_fails],
                    "soft_fails": [c.name for c in report.soft_fails],
                }
            )
        )
        provenance.finish()

        return OperatorResult(outputs={"report": report}, provenance=provenance)
