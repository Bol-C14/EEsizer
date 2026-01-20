# 2026-01-20 16:00 — LLM Policy Layering Fix

## Scope

- Move LLM tool invocation out of `LLMPatchPolicy` and into `PatchLoopStrategy`.
- Keep LLMPatchPolicy pure (prompt formatting + response parsing only).
- Preserve LLM audit artifacts and retry behavior via strategy-managed calls.

## Files touched

- `src/eesizer_core/policies/llm_patch.py`
- `src/eesizer_core/strategies/patch_loop.py`
- `docs/policies/llm_patch.md`
- `docs/specs/41_policy_strategy_contracts.md`
- `tests/test_m1_llm_patch_policy_retry.py`
- `tests/test_m1_mock_patch_clamp.py`

## Rationale & notes

- Restores strict layering: policies no longer execute tools or write run artifacts.
- LLM calls remain operator-backed and fully recorded via provenance.

## Migration / compatibility

- `LLMPatchPolicy.propose()` now returns `Patch(stop=True, notes="llm_requires_strategy")`.
- Strategies must use `LLMPatchPolicy.build_request()` and `parse_response()` to integrate LLMs.
