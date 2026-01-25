# 2026-01-15 16:30 — Step 8 objective alias fix

## Summary
- Injected a minimal .print directive into legacy decks to avoid ngspice 39.x nonzero exits.
- Objective evaluation now falls back to canonicalized metric names.
- Added a unit test to lock objective alias behavior.

## Files touched
- Code: `src/eesizer_core/baselines/legacy_metrics_adapter.py`, `src/eesizer_core/analysis/objective_eval.py`
- Tests: `tests/test_step8_objective_alias_eval.py`

## Notes / Rationale
- Prevents legacy baseline from failing due to ngspice batch return codes.
- Ensures objective evaluation remains correct when metrics are canonicalized.
