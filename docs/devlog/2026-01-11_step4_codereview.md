# Code Review: Step4 Sim Stack & Metrics

## Scope
- DeckBuildOperator (ac/dc/tran control injection)
- NgspiceRunOperator (batch run + RawSimData)
- MetricRegistry + ComputeMetricsOperator (ac/dc/tran metrics)
- Examples and fixtures

## Functional Acceptance
- ✅ DeckBuildOperator generates AC decks with `.control` and writes `ac.csv` (tests cover AC/DC/TRAN templates).
- ✅ NgspiceRunOperator wraps ngspice, stages outputs under `RunContext.run_dir/<stage>/...`, returns RawSimData (cmdline/log path/returncode).
- ✅ MetricRegistry + ComputeMetricsOperator compute AC metrics (defaults include `ac_mag_db_at_1k`, `ac_unity_gain_freq`; DC/TRAN added).
- ✅ All outputs live under `RunContext.run_dir` (runner uses ctx.run_dir; no hardcoded paths).

## Engineering Checks
- Purity: metric functions are pure (read CSV only; no writes or global paths).
- Error surfaces: runner raises SimulationError with returncode/log tail/cmdline; ComputeMetricsOperator propagates diagnostics.
- Tests: deck builder template correctness, runner with mocked subprocess, metrics via fixtures (ac/dc/tran) — `PYTHONPATH=src pytest -q` passes.
- Legacy isolation: no runtime imports from `legacy/`; only migrated logic.

## Extensibility
- Adding a metric requires only a compute function + MetricImplSpec registration (registry/operator are generic).
- Deck/run/metrics are decoupled (no strategy/runner edits needed for new metrics).

## Notes / Risks
- DC/TRAN metrics are minimal; additional specs (overshoot, currents) still needed.
- Examples focus on AC; DC/TRAN example manifests could further validate staging with real ngspice.

## Verification
- Tests executed: `PYTHONPATH=src pytest -q` (all green).
