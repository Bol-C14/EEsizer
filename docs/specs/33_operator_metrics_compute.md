# 33. Operator: ComputeMetricsOperator

**Code:** `eesizer_core.metrics.operators.ComputeMetricsOperator`

## Purpose

Compute a set of metrics from `RawSimData` using the metric registry.

## Inputs

- `raw_data: RawSimData` (required)
- `metric_names: Sequence[str] | str` (required)
  The operator resolves these names via the registry.

## Outputs

- `metrics: MetricsBundle`

## Registry behavior

The operator uses `MetricRegistry`:
- metric names must be unique
- metrics declare `requires_kind` and `requires_outputs`
- missing required outputs MUST raise `ValidationError`

## Failure modes

- `ValidationError` if input types are wrong
- `MetricError` if a requested metric does not exist, is incompatible, or computation fails

## Provenance

Recommended entries:
- raw data fingerprint (kind + output paths + return code)
- metric names list
- resulting metric values dictionary hash
