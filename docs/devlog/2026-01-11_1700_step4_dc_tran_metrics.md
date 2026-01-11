# DevLog: Step4 DC/TRAN Decks and Metrics

- Date: 2026-01-11 17:00 (UTC)
- Scope: Extend deck builder to DC/TRAN outputs and add initial DC/TRAN metrics + fixtures.

## Goals

1. Generate consistent CSV outputs for dc/tran analyses alongside ac.
2. Provide at least one usable metric per analysis, with diagnostics when computation is not possible.
3. Keep metrics pure: read existing outputs only, no path assembly in metric functions.

## Work Completed

- `DeckBuildOperator` now supports `SimKind.dc` and `SimKind.tran`:
  - DC: `dc <source> start stop step` with `wrdata dc.csv v(<sweep_node>) v(<outputs...>)`
  - TRAN: `tran <step> <stop>` with `wrdata tran.csv time v(<outputs...>)`
  - Still deterministic, control-free source netlist preserved.
- Added DC/TRAN metrics:
  - DC: `compute_dc_vout_last`, `compute_dc_slope` (returns None + diagnostics on insufficient data)
  - TRAN: `compute_tran_rise_time` (10–90% rise time, can return None with reason)
  - Defaults registered in `metrics/defaults.py`.
- Enhanced `ComputeMetricsOperator` to accept compute functions that return `(value, diagnostics)` and attach diagnostics to `MetricValue.details`.
- Fixtures/tests:
  - New fixtures `tests/fixtures/dc.csv`, `tests/fixtures/tran.csv`
  - Expanded `tests/test_metrics_algorithms.py` to cover dc/tran metrics and operator behavior.

## Notes / Next Steps

- Metrics remain minimal; more DC/TRAN metrics (overshoot, fall time, current probes) can be added later.
- Integration demo for dc/tran can reuse `DeckBuildOperator` + `NgspiceRunOperator` once corresponding netlists/SimPlans are provided.
- Keep `wrdata` column naming stable (`frequency`, `v(...)`, `time`) to avoid glue in future metrics.
