# DevLog: Step4 Metric Registry + AC Metrics

- Date: 2026-01-11 14:00 (UTC)
- Scope: Introduce metric registry, AC metric functions, compute operator, fixtures, and example integration.

## Goals

1. Make metrics explicit and discoverable via a registry instead of ad-hoc functions.
2. Keep metric computations pure (read existing outputs only) and analysis-kind aware.
3. Provide fixtures and an example to exercise AC metrics end-to-end.

## Work Completed

- Added `MetricImplSpec`/`MetricRegistry` (`src/eesizer_core/metrics/registry.py`) and default registry (`metrics/defaults.py`) with AC metrics:
  - `ac_mag_db_at_1k` (gain at 1 kHz, node `out`)
  - `ac_unity_gain_freq` (first 0 dB crossing for node `out`)
- Implemented AC metric functions (`src/eesizer_core/metrics/ac.py`) and a `ComputeMetricsOperator` (`metrics/operators.py`) that maps `RawSimData` + metric names to `MetricsBundle`.
- Added tests and fixtures:
  - `tests/fixtures/ac.csv` sample data
  - `tests/test_metrics_algorithms.py` covering function outputs, operator registry handling, and error paths.
- Updated example (`examples/run_ac_once.py`) to compute metrics after running ngspice.

## Notes / Next Steps

- Registry currently covers AC-only metrics; DC/TRAN metrics to be added with corresponding deck outputs.
- Metrics expect `ac.csv` produced by `wrdata` in the deck builder; keep column naming aligned (`frequency`, `vdb(node)`, `vp(node)`).
- Consider adding tolerance handling and richer diagnostics in `MetricValue.details` once more specs are migrated.
