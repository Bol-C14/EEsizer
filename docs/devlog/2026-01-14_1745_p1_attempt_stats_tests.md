# 2026-01-14 17:45 — P1 attempt stats and tests

## Summary
- Added sim run OK/failed counters and recorded them in summary output.
- Streamed history loading to avoid full-file reads.
- Added tests for attempt-stage naming and failed sim run counts.

## Files touched
- Code: `src/eesizer_core/strategies/patch_loop/strategy.py`, `src/eesizer_core/runtime/run_loader.py`, `src/eesizer_core/contracts/provenance.py`
- Docs: `docs/specs/11_runtime_layout.md`, `docs/specs/25_artifact_provenance.md`
- Tests: `tests/test_step7_stage_attempt_names.py`, `tests/test_step7_sim_runs_failed_counts.py`

## Notes / Rationale
- Makes simulation cost accounting more faithful for comparative studies.
- Prevents future regressions in attempt-stage naming and run accounting.
