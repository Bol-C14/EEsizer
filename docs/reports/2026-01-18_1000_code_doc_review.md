# EEsizer Code and Documentation Review

Date: 2026-01-17

## Scope
- Code: `src/eesizer_core` (contracts, domain/spice, operators, strategies, metrics, runtime, analysis)
- Tests: `tests/`
- Docs: `README.md`, `docs/wiki`, `docs/specs`
- Examples: `examples/`

Tests were not executed as part of this review.

## Summary
- Findings: 1 high, 2 medium, 3 low (docs)
- Primary risks: patch validation crash on non-numeric W/L values, silent duplicate ParamSpace entries, incomplete objective reporting in run comparisons

## Findings

### High
1. W/L ratio validation can raise exceptions on non-numeric existing values.
   - Location: `src/eesizer_core/domain/spice/patching.py:122-140`
   - Details: When `wl_ratio_min` is set, `validate_patch` calls `_parse_scalar_numeric` directly on current W/L text. If the netlist uses parameterized or expression-based W/L values (common in SPICE), `_parse_scalar_numeric` raises `ValidationError` and the guard pipeline crashes instead of returning a structured `PatchValidationResult`.
   - Impact: Patch flows can fail hard when W/L values are non-numeric, even if the patch itself is otherwise valid. This blocks optimization for parameterized netlists.
   - Recommendation: Wrap W/L parsing in `try/except` and convert failures into `errors`/`ratio_errors` entries (or skip ratio checks for non-numeric values). Add a regression test covering non-numeric W/L values with `wl_ratio_min` enabled.

### Medium
2. `ParamSpace.build` silently overwrites duplicate `param_id`s.
   - Location: `src/eesizer_core/contracts/artifacts.py:105-108`
   - Details: The index is built with a dict comprehension and does not detect duplicates.
   - Impact: Duplicate parameter definitions are silently dropped, leading to inconsistent bounds/frozen flags and unpredictable patch behavior.
   - Recommendation: Detect duplicates and raise `ValidationError` (or keep first and log warnings). Add a unit test for duplicate ids.

3. Run comparison report drops objectives when lists differ.
   - Location: `src/eesizer_core/analysis/compare_runs.py:214-309`
   - Details: The report uses `zip(objectives_a, objectives_b)` to render the objectives table. If the objective lists differ in length or ordering, extra entries are omitted.
   - Impact: The report can silently hide mismatched objectives, which undermines comparison fidelity.
   - Recommendation: Render objectives keyed by metric name and include missing entries with explicit "missing" markers. Add a test where objective counts differ.

### Low (Docs)
4. Policy/strategy signature mismatch between docs and implementation.
   - Docs: `docs/specs/41_policy_strategy_contracts.md:13-44`
   - Code: `src/eesizer_core/contracts/policy.py:13-28`, `src/eesizer_core/contracts/strategy.py:20-40`
   - Details: Docs show `Policy.propose(observation: Mapping[str, Any])` and `Strategy.run(source, spec, ctx, **cfg)`. Code expects `Observation` and uses `(spec, source, ctx, cfg)` ordering.
   - Impact: Integrators may implement incompatible policies/strategies based on docs.
   - Recommendation: Update docs to reflect actual signatures and parameter ordering.

5. Project status and repo layout docs are outdated.
   - Docs: `README.md:21-25`, `docs/wiki/01_repo_layout.md:31-46`
   - Details: Both documents describe strategies/policies as "next" or "planned" even though they are implemented in `src/eesizer_core/strategies` and `src/eesizer_core/policies`.
   - Impact: Onboarding confusion; contributors may not discover existing implementations.
   - Recommendation: Update the status and layout sections to reflect current code.

6. PatchApplyOperator provenance spec not met.
   - Docs: `docs/specs/34_operator_patch_apply.md:43-49`
   - Code: `src/eesizer_core/operators/netlist/patch_apply.py:55-105`
   - Details: Spec requires provenance to include topology signatures before/after, but implementation only records patched outputs.
   - Impact: Audit trail is weaker than the spec suggests.
   - Recommendation: Either update spec or record signature_before/signature_after in provenance (and/or outputs).

## Testing Gaps / Recommendations
- Add a test for `validate_patch` with `wl_ratio_min` and non-numeric W/L values (to verify errors are returned instead of exceptions).
- Add multi-output-node coverage for `DeckBuildOperator` + `load_wrdata_table` to validate expected column ordering.
- Add tests for duplicate `ParamSpace` ids and for objective list mismatches in `compare_runs`.

## Open Questions / Assumptions
- Confirm ngspice `wrdata` column ordering for multi-node outputs (single frequency/time column vs repeated). The current metadata assumes repeated scale columns with suffixes; this should be verified against ngspice behavior.
