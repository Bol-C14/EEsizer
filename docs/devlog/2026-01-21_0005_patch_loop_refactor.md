# 2026-01-21 00:05 — PatchLoop Strategy Refactor

## Scope

- Split PatchLoopStrategy into focused modules (planning, evaluate, attempt, state) without changing behavior.
- Keep strategy orchestration readable and preserve existing run/recording outputs.

## Files touched

- `src/eesizer_core/strategies/patch_loop/strategy.py`
- `src/eesizer_core/strategies/patch_loop/state.py`
- `src/eesizer_core/strategies/patch_loop/planning.py`
- `src/eesizer_core/strategies/patch_loop/evaluate.py`
- `src/eesizer_core/strategies/patch_loop/attempt.py`
- `src/eesizer_core/strategies/patch_loop/__init__.py`
- `src/eesizer_core/strategies/baseline_noopt.py`
- `src/eesizer_core/strategies/patch_loop.py` (removed)
- `src/eesizer_core/runtime/context.py`

## Rationale & notes

- Moves policy proposal, attempt execution, metrics evaluation, and state tracking into their own files to match
  the Operator/Policy/Strategy split and improve testability.
- Preserves stage naming, retry behavior, guard handling, and run recording semantics.
- Restores `RunContext.recorder()` so strategies can record manifests and summaries as intended.

## Migration / compatibility

- `PatchLoopStrategy` remains available from `eesizer_core.strategies.patch_loop`.
- Legacy helper names for metric grouping/planning are aliased in the package init to avoid import breakage.
