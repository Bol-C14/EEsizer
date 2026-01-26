# 2026-01-26 1645 Step4 plots + report insights

- Added deterministic plot data extraction and rendering pipeline under `analysis/plotting`.
- Added ReportPlotsOperator to generate plot data/PNGs, update `report.md`, and register plot artifacts in the manifest.
- Grid and corner strategies now call the plot operator and emit `plots/index.json` plus plot artifacts.
- Added sensitivity insights (Spearman) with `insights/sensitivity.json` and report embedding.
- Added Step4 unit tests for plot data and mock plot generation for grid/corner runs.
- Updated runtime layout/spec docs to list `plots/` and insights artifacts.

Usage:
- Grid search plots: `PYTHONPATH=src python examples/run_grid_search_bench.py --bench ota --max-iters 21 --levels 7 --span-mul 4 --scale log`
- Unit tests: `pytest -q tests/test_step4_plot_data_heatmap_unit.py tests/test_step4_plot_data_tradeoff_unit.py tests/test_step4_plot_index_determinism_unit.py tests/test_step4_plots_generated_grid_mock.py tests/test_step4_plots_generated_corner_mock.py`

Caveats:
- If matplotlib is missing, plots are skipped with a recorded reason in `plots/index.json`.
