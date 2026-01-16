# 2026-01-14 12:00 — Step 6 guard system

## Summary
- Added GuardCheck/GuardReport artifacts plus guard operators (patch/topology/behavior/chain, formal stub).
- Integrated guards into PatchLoopStrategy with retry-on-guard/sim/metric failures and guard reporting in history.
- Added Step 6 unit tests for guard logic, constraints, and strategy retries.

## Files touched
- Code: `src/eesizer_core/contracts/guards.py`, `src/eesizer_core/operators/guards/*`, `src/eesizer_core/strategies/patch_loop.py`
- Docs: `docs/specs/26_artifact_guards.md`, `docs/specs/35_operator_guards.md`, `docs/specs/20_artifact_contracts.md`, `docs/specs/41_policy_strategy_contracts.md`, `docs/specs/00_index.md`
- Tests: `tests/test_step6_guard_contract_smoke.py`, `tests/test_step6_patch_guard.py`, `tests/test_step6_topology_guard.py`, `tests/test_step6_behavior_guard.py`, `tests/test_step6_constraints_in_spec.py`, `tests/test_step6_strategy_retry_on_guard_fail.py`, `tests/test_step6_strategy_handles_sim_error.py`

## Notes / Rationale
- Guards are now explicit operators with structured outputs, enabling auditing and future plug-ins (formal/coverage).
- Patch guards enforce spec constraints and numeric sanity before netlist mutation.
- Behavior guard checks metrics and scans logs for fatal patterns without crashing the strategy loop.
