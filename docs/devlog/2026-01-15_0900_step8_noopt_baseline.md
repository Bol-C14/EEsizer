# 2026-01-15 09:00 — Step 8 NoOpt baseline

## Summary
- Added `NoOptBaselineStrategy` to run a single baseline simulation with run artifacts.
- Introduced a `baseline_noopt` stop reason for baseline runs.
- Added an example entry point and a unit test for baseline outputs.

## Files touched
- Code: `src/eesizer_core/strategies/baseline_noopt.py`, `src/eesizer_core/strategies/__init__.py`, `src/eesizer_core/contracts/enums.py`
- Tests: `tests/test_step8_baseline_noopt.py`
- Examples: `examples/run_noopt_baseline.py`
- Docs: `docs/specs/25_artifact_provenance.md`

## Notes / Rationale
- Baseline runs are required for clean comparisons against optimization strategies.
- Artifacts reuse the Step 7 recorder/manifest to keep outputs auditable.

## Migration notes
- New stop reason `baseline_noopt` appears in run summaries for baseline runs.
