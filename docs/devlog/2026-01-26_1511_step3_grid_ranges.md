# 2026-01-26 1511 Step3 grid ranges + deterministic candidates

- Added grid search range tracing (RangeTrace) with bounds/nominal span rules and log fallback warnings.
- Added deterministic candidate generation metadata (`search/candidates_meta.json`) and persisted ranges (`search/ranges.json`).
- Grid search report now includes parameter selection, ranges/discretization, candidate generation, TopK deltas vs nominal, and failure breakdown.
- Added benchmark grid search runner that reads `bench.json` recommended knobs (`examples/run_grid_search_bench.py`).
- Added unit tests for range inference and candidate truncation determinism plus a mock report check.
- Updated runtime/layout and strategy docs to list new grid search artifacts.

Usage:
- Bench grid search: `PYTHONPATH=src python examples/run_grid_search_bench.py --bench ota --max-iters 21 --levels 7 --span-mul 4 --scale log`
- Unit tests: `pytest -q tests/test_step3_ranges_unit.py tests/test_step3_candidate_generation_unit.py tests/test_step3_grid_report_contains_ranges_mock.py`

Caveats:
- Explicit `param_ids` that include frozen params still require `allow_param_ids_override_frozen=true`.
- If a param has no bounds and no numeric nominal, it is skipped with a recorded reason.
