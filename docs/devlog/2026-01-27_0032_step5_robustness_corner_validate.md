# 2026-01-27 0032 Step5 robustness corner-validate

- Added `CornerValidateOperator` to validate grid candidates under deterministic corners and write:
  - `search/corner_set.json`
  - `search/robust_candidates.json`
  - `search/robust_topk.json`
  - `search/robust_pareto.json`
  - `search/robust_meta.json`
- Plotting now supports grid+corner_validate runs without rewriting history: `extract_plot_context()` falls back to `search/robust_*.json` and enables `plots/robust_nominal_vs_worst.png`.
- Report upgrades:
  - Inserts `## Robustness Validation` section into `report.md` (corner config + robust Top-K table + paper-friendly comparison sentence when available).
  - Re-runs `ReportPlotsOperator` after corner validation to embed robustness plots.
- Corner cost-control modes supported: `global`, `oat`, `oat_topm` (uses `insights/sensitivity.json` when present).
- Added example entrypoints:
  - `examples/run_bench_grid_then_corner_validate.py`
  - `examples/run_bench_grid_then_corner_validate_ota.py`
  - `examples/run_bench_grid_then_corner_validate_opamp3.py`
- Tests added for DoD (mock, no ngspice required): `tests/test_step5_*`.

Quick checks:
- `pytest -q`
- `pytest -q -m integration`
- `PYTHONPATH=src python examples/run_bench_grid_then_corner_validate_ota.py --max-iters 21 --levels 7 --span-mul 4 --scale log --mode coordinate --continue-after-baseline-pass --corners oat`

