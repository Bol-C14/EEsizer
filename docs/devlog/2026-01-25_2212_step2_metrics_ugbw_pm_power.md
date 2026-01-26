# 2026-01-25 2212 Step2 metrics (UGBW/PM/Power)

- Added canonical metric names/definitions in `contracts/metrics.py` and wired UGBW/PM/Power into the default registry.
- Implemented deterministic open-loop AC metrics (UGBW + PM) with log-domain interpolation and explicit missing reasons.
- Implemented DC power metric from `abs(i(VDD)) * VDD` and added missing-probe diagnostics.
- Updated OTA/opamp3 specs to use UGBW/PM/Power objectives and to probe v(vdd)/i(VDD).
- Reports (baseline/grid/corner/orchestrator) now include metric definitions and missing-status lines.
- Updated metrics docs and added unit/integration tests for Step2.

Usage:
- Baseline with updated benchmarks: `PYTHONPATH=src python examples/run_benchmark_baseline.py --bench ota --out examples/output`
- Optional integration test (ngspice required): `pytest -q -m integration tests/test_step2_bench_metrics_integration_ngspice.py`

Caveats:
- If AC sweep never crosses 0 dB, UGBW/PM are reported as missing with a reason.
