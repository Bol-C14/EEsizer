# 2026-01-15 11:30 — Step 8 metric aliases

## Summary
- Added canonical metric aliasing to unify legacy and new naming.
- Baseline outputs now canonicalize best metrics before writing `best_metrics.json`.

## Files touched
- Code: `src/eesizer_core/metrics/aliases.py`, `src/eesizer_core/metrics/__init__.py`, `src/eesizer_core/strategies/baseline_noopt.py`, `src/eesizer_core/baselines/legacy_metrics_adapter.py`, `src/eesizer_core/strategies/patch_loop.py`
- Tests: `tests/test_step8_metric_aliases.py`

## Notes / Rationale
- Canonical names reduce drift between legacy and new metrics pipelines.
- Canonicalization is applied at baseline output stages to keep artifacts comparable.
