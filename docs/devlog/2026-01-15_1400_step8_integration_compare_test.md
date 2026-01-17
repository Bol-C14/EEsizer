# 2026-01-15 14:00 — Step 8 integration compare test

## Summary
- Added an integration test that runs NoOpt and legacy baselines on the same netlist and compares outputs.

## Files touched
- Tests: `tests/test_step8_compare_new_vs_legacy_metrics_integration.py`

## Notes / Rationale
- Locks the new vs legacy metrics comparison into CI to catch regressions.
