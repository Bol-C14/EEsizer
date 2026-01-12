from __future__ import annotations

from typing import Any, Iterable, Mapping

from ..contracts.errors import MetricError, ValidationError
from ..contracts.operators import Operator, OperatorResult
from ..contracts.provenance import ArtifactFingerprint, Provenance, stable_hash_json
from ..contracts.artifacts import MetricValue, MetricsBundle
from ..sim.artifacts import RawSimData
from .registry import MetricRegistry
from .defaults import DEFAULT_REGISTRY


class ComputeMetricsOperator(Operator):
    """Compute metrics from RawSimData using a MetricRegistry."""

    name = "compute_metrics"
    version = "0.1.0"

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY

    def run(self, inputs: Mapping[str, Any], ctx: Any) -> OperatorResult:
        raw = inputs.get("raw_data")
        if not isinstance(raw, RawSimData):
            raise ValidationError("raw_data must be provided as RawSimData")

        metric_names = inputs.get("metric_names")
        if metric_names is None:
            raise ValidationError("metric_names must be provided")
        if isinstance(metric_names, str):
            metric_names = [metric_names]
        if not isinstance(metric_names, Iterable):
            raise ValidationError("metric_names must be an iterable of strings")

        specs = self.registry.resolve(metric_names)
        metrics = MetricsBundle()
        provenance = Provenance(operator=self.name, version=self.version)
        provenance.inputs["raw_data"] = ArtifactFingerprint(
            sha256=stable_hash_json(
                {
                    "kind": raw.kind.value,
                    "outputs": {k: str(v) for k, v in raw.outputs.items()},
                    "log_path": str(raw.log_path),
                    "run_dir": str(raw.run_dir),
                }
            )
        )
        provenance.inputs["metric_names"] = ArtifactFingerprint(sha256=stable_hash_json(list(metric_names)))

        for spec in specs:
            if raw.kind != spec.requires_kind:
                raise ValidationError(
                    f"Metric '{spec.name}' requires {spec.requires_kind.value} data, got {raw.kind.value}"
                )
            for out_name in spec.requires_outputs:
                if out_name not in raw.outputs:
                    raise ValidationError(f"Metric '{spec.name}' requires output '{out_name}'")

            try:
                value, diag = spec.compute_fn(raw, spec)
            except Exception as exc:
                raise MetricError(f"Failed to compute metric '{spec.name}': {exc}") from exc

            details = dict(spec.params)
            if isinstance(diag, dict):
                details.update(diag)
            metrics.values[spec.name] = MetricValue(name=spec.name, value=value, unit=spec.unit, details=details)

        provenance.outputs["metrics"] = ArtifactFingerprint(
            sha256=stable_hash_json({k: v.value for k, v in metrics.values.items()})
        )
        provenance.finish()

        return OperatorResult(outputs={"metrics": metrics}, provenance=provenance)
