# 24. Metrics Artifacts

Metrics translate simulator evidence into comparable numbers.

## 24.1 MetricSpec

**Type:** `eesizer_core.contracts.artifacts.MetricSpec`

Fields:
- `name: str` (unique key)
- `description: str`
- `unit: str` (free-form)
- `sim_kind: SimKind | None`  
  If set, the metric requires evidence from that simulation kind.

## 24.2 MetricValue

**Type:** `eesizer_core.contracts.artifacts.MetricValue`

Fields:
- `name: str`
- `value: float`
- `unit: str`
- `meta: dict[str, Any] | None`

`meta` can include uncertainties, fit quality, corner labels, etc.

## 24.3 MetricsBundle

**Type:** `eesizer_core.contracts.artifacts.MetricsBundle`

Fields:
- `values: tuple[MetricValue, ...]`

Convenience method:
- `MetricsBundle.as_dict() -> dict[str, float]`

## 24.4 Implementation specs in the registry

The metric registry uses an internal implementation spec:
- `eesizer_core.metrics.registry.MetricImplSpec`

Fields:
- `spec: MetricSpec`
- `compute: (RawSimData) -> MetricValue`

The registry enforces a single global name per metric.

