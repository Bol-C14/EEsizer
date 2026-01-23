# 2026-01-16 10:00 — Greedy coordinate policy

- Scope: add a coordinate-descent/hill-climb policy with adaptive steps and guard-aware retries.
- Code: `src/eesizer_core/policies/greedy_coordinate.py`, `src/eesizer_core/policies/__init__.py`, `src/eesizer_core/api.py`, `src/eesizer_core/__init__.py`
- Strategy updates: `src/eesizer_core/strategies/patch_loop/strategy.py` adds `current_score`, `best_score`, `param_values`, and `last_guard_report` to policy observations.
- Domain helper: `src/eesizer_core/domain/spice/patching.py` adds `extract_param_values`.
- Tests: `tests/test_policy_greedy_coordinate_unit.py`, `tests/test_policy_greedy_coordinate_bounds.py`, `tests/test_policy_greedy_coordinate_rejection.py`, `tests/test_policy_greedy_coordinate_integration.py`
- Examples: `examples/run_patch_loop_greedy.py`, `examples/circuits/rc_lowpass.sp`, `examples/specs/rc_lowpass.json`
- Docs: `docs/specs/41_policy_strategy_contracts.md`, `docs/policies/greedy_coordinate.md`

Notes:
- Policy remains patch-only and does not call operators directly.
- Observation notes are used to pass score/param snapshots without changing the core Observation contract.
