# 2026-01-20 17:00 — PatchLoop Baseline Uses Sanitized Netlist

## Scope

- Ensure PatchLoop baseline and subsequent simulations use the sanitized netlist.
- Capture DeckBuildOperator validation failures in baseline guard handling.

## Files touched

- `src/eesizer_core/strategies/patch_loop.py`

## Rationale & notes

- The topology signature already sanitizes the netlist; simulations now consume the same sanitized text to avoid
  unsafe includes or `.control` blocks affecting baseline metrics.
- Baseline deck-build validation errors are now converted into guard failures rather than crashing the run.

## Migration / compatibility

- Runs record the sanitized netlist as `inputs/source.sp` when sanitization changes the text.
