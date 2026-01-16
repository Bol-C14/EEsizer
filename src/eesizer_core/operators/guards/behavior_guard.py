from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

from ...contracts.artifacts import CircuitSpec, MetricsBundle
from ...contracts.errors import ValidationError
from ...contracts.guards import GuardCheck
from ...contracts.operators import Operator, OperatorResult
from ...contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json


_DEFAULT_HARD_PATTERNS = (
    r"fatal",
    r"panic",
    r"abort",
    r"singular matrix",
)
_DEFAULT_SOFT_PATTERNS = (
    r"timestep too small",
    r"warning",
)


def _compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pat, re.IGNORECASE) for pat in patterns]


class BehaviorGuardOperator(Operator):
    """Post-sim guard to catch missing metrics and suspicious log patterns."""

    name = "behavior_guard"
    version = "0.1.0"

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        metrics = inputs.get("metrics")
        if not isinstance(metrics, MetricsBundle):
            raise ValidationError("BehaviorGuardOperator: 'metrics' must be MetricsBundle")
        spec = inputs.get("spec")
        if not isinstance(spec, CircuitSpec):
            raise ValidationError("BehaviorGuardOperator: 'spec' must be CircuitSpec")
        stage_map = inputs.get("stage_map") or {}
        if not isinstance(stage_map, Mapping):
            raise ValidationError("BehaviorGuardOperator: 'stage_map' must be a mapping")
        guard_cfg = dict(inputs.get("guard_cfg") or {})

        hard_reasons: list[str] = []
        soft_reasons: list[str] = []

        # Required objective metrics must exist and be finite.
        for obj in spec.objectives:
            metric = metrics.get(obj.metric)
            if metric is None:
                hard_reasons.append(f"metric '{obj.metric}' missing")
                continue
            value = metric.value
            if value is None:
                hard_reasons.append(f"metric '{obj.metric}' is None")
                continue
            if not math.isfinite(value):
                hard_reasons.append(f"metric '{obj.metric}' is non-finite")

        if guard_cfg.get("scan_logs", True):
            hard_patterns = guard_cfg.get("log_hard_patterns", list(_DEFAULT_HARD_PATTERNS))
            soft_patterns = guard_cfg.get("log_soft_patterns", list(_DEFAULT_SOFT_PATTERNS))
            soft_as_hard = guard_cfg.get("soft_log_as_hard", False)

            hard_regexes = _compile_patterns(list(hard_patterns))
            soft_regexes = _compile_patterns(list(soft_patterns))

            for kind, run_dir in stage_map.items():
                kind_name = getattr(kind, "value", str(kind))
                log_path = Path(run_dir) / f"ngspice_{kind_name}.log"
                if not log_path.exists():
                    soft_reasons.append(f"missing log '{log_path}'")
                    continue
                text = log_path.read_text(encoding="utf-8", errors="ignore")
                for regex in hard_regexes:
                    if regex.search(text):
                        hard_reasons.append(f"log '{log_path.name}' matched '{regex.pattern}'")
                for regex in soft_regexes:
                    if regex.search(text):
                        msg = f"log '{log_path.name}' matched '{regex.pattern}'"
                        if soft_as_hard:
                            hard_reasons.append(msg)
                        else:
                            soft_reasons.append(msg)

        if hard_reasons:
            ok = False
            severity = "hard"
            reasons = tuple(hard_reasons + soft_reasons)
        elif soft_reasons:
            ok = False
            severity = "soft"
            reasons = tuple(soft_reasons)
        else:
            ok = True
            severity = "hard"
            reasons = ()

        check = GuardCheck(
            name="behavior_guard",
            ok=ok,
            severity=severity,  # type: ignore[arg-type]
            reasons=reasons,
            data={"hard_reasons": hard_reasons, "soft_reasons": soft_reasons},
        )

        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["metrics"] = ArtifactFingerprint(
            sha256=stable_hash_json({k: v.value for k, v in metrics.values.items()})
        )
        provenance.inputs["stage_map"] = ArtifactFingerprint(
            sha256=stable_hash_json({str(k): str(v) for k, v in stage_map.items()})
        )
        provenance.outputs["check"] = ArtifactFingerprint(
            sha256=stable_hash_json({"ok": check.ok, "severity": check.severity, "reasons": list(check.reasons)})
        )
        provenance.finish()

        return OperatorResult(outputs={"check": check}, provenance=provenance)
