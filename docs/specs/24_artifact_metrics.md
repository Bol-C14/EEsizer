# 24. Metrics Artifacts

Metrics translate simulator evidence into comparable numbers.

## 24.1 MetricSpec

**Type:** `eesizer_core.contracts.artifacts.MetricSpec`

Fields:
- `name: str` (unique key)
- `unit: str`
- `sim: SimKind` (required simulation kind)
- `required_files: tuple[str, ...]` (logical output names, e.g., `"ac_csv"`)
- `description: str`
- `params: dict[str, Any]`

## 24.2 MetricValue

**Type:** `eesizer_core.contracts.artifacts.MetricValue`

Fields:
- `name: str`
- `value: float | None`
- `unit: str`
- `passed: bool | None`
- `details: dict[str, Any]`

## 24.3 MetricsBundle

**Type:** `eesizer_core.contracts.artifacts.MetricsBundle`

Fields:
- `values: dict[str, MetricValue]`

Helpers:
- `get(name)` -> `MetricValue | None`

## 24.4 Implementation specs in the registry

The metric registry uses an implementation spec:
- `eesizer_core.metrics.registry.MetricImplSpec`

Fields:
- `name: str`
- `unit: str`
- `requires_kind: SimKind`
- `requires_outputs: tuple[str, ...]`
- `compute_fn: Callable[[RawSimData, MetricImplSpec], tuple[float | None, dict]]`
- `params: dict[str, Any]`
- `description: str`

The registry enforces a single global name per metric (registry key must match `spec.name`).

## 24.5 Canonical metric definitions

Step2 introduces a shared definition list in `eesizer_core.contracts.metrics`:
- `UGBW_HZ`, `PHASE_MARGIN_DEG`, `POWER_W`
- `MetricDefinition` + `METRIC_DEFINITIONS`

Reports SHOULD include the definition text for any metrics they list.

## 24.6 Missing data semantics

If a metric cannot be computed (missing outputs, missing probes, no unity crossing), it should:
- return `value: null`
- set `details.status = "missing"` and a `details.reason`

Strategies and reports SHOULD surface these reasons instead of crashing.
