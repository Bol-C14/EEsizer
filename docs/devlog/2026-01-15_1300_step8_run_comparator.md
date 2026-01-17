# 2026-01-15 13:00 — Step 8 run comparator

## Summary
- Added `compare_runs` to generate `comparison.json` and `report.md` from two run dirs.
- Added default metric tolerances and canonicalization in the comparison flow.
- Added a minimal CLI entrypoint `eesizer compare`.

## Files touched
- Code: `src/eesizer_core/analysis/compare_runs.py`, `src/eesizer_core/analysis/__init__.py`, `src/eesizer_core/cli.py`, `src/eesizer_core/metrics/tolerances.py`, `src/eesizer_core/metrics/__init__.py`, `pyproject.toml`
- Tests: `tests/test_step8_compare_runs.py`

## Notes / Rationale
- Comparison outputs are deterministic JSON + human-readable markdown for reporting.
- Tolerances live in metrics to keep comparison logic thin and testable.
