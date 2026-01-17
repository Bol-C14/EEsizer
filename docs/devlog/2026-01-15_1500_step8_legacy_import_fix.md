# 2026-01-15 15:00 — Step 8 legacy import + integration netlist fix

## Summary
- Added robust legacy directory discovery and import wiring for legacy baselines.
- Updated integration test netlist to include a DC path and aligned legacy import checks.
- Canonicalized objective metric names during comparison to avoid mismatch after aliasing.

## Files touched
- Code: `src/eesizer_core/baselines/legacy_metrics_adapter.py`, `src/eesizer_core/metrics/aliases.py`, `src/eesizer_core/metrics/__init__.py`, `src/eesizer_core/analysis/compare_runs.py`
- Tests: `tests/test_step8_compare_new_vs_legacy_metrics_integration.py`

## Notes / Rationale
- Legacy adapter now locates `<repo>/legacy/legacy_eesizer` without relying on PYTHONPATH.
- Integration test avoids DC singularities by adding a high-value leakage resistor.
