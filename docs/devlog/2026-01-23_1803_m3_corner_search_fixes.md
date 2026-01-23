# 2026-01-23 18:03 — Corner Search Audit Fixes

## Summary
- Refactored corner search into a package and aligned behavior with audit findings.
- Centralized attempt execution and sim run accounting across patch/grid/corner strategies.
- Improved manifest indexing, artifact replay support, and documentation coverage.

## Scope
- Corner search now defaults to per-parameter OAT corners with optional global corners.
- Corner overrides are relative by default and baseline corner failures do not gate search unless configured.
- Search artifacts are indexed in manifests, and orchestrator loads best results via RunLoader.
- ArtifactStore persists type tags and can rehydrate core dataclasses from disk.
- Added tests for corner defaults, sim run counts, manifest indexing, and frozen param filtering.

## Files touched
- Strategy pipeline: `src/eesizer_core/strategies/attempt_pipeline.py`, `src/eesizer_core/strategies/grid_search.py`, `src/eesizer_core/strategies/patch_loop/evaluate.py`.
- Corner search package: `src/eesizer_core/strategies/corner_search/strategy.py`, `src/eesizer_core/strategies/corner_search/measurement.py`, `src/eesizer_core/search/corners.py`.
- Runtime: `src/eesizer_core/runtime/recording_utils.py`, `src/eesizer_core/runtime/artifact_store.py`, `src/eesizer_core/strategies/multi_agent_orchestrator.py`.
- Tests: `tests/test_m3_corner_generator_unit.py`, `tests/test_m3_corner_search_outputs_mock.py`, `tests/test_sim_runs_accounting_multi_kind.py`, `tests/test_param_ids_filter_frozen.py`.
- Docs/examples/packaging: `README.md`, `docs/specs/11_runtime_layout.md`, `docs/specs/41_policy_strategy_contracts.md`, `docs/wiki/02_dev_workflow.md`, `examples/run_corner_search_rc.py`, `MANIFEST.in`, `pyproject.toml`.

## Notes
- Corner stages include a corner tag to avoid stage collisions.
- Run manifests now capture search outputs under `search/`.
