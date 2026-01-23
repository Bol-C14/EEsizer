# 2026-01-14 16:30 — P0 fixes (ngspice resolver, attempt stages, sim run accounting)

## Summary
- Hardened ngspice resolver to ignore non-runnable binaries (wrong arch/vendor).
- Added attempt suffix to stage directories to prevent overwrite across retries.
- Counted simulation runs for all attempts (success or guard-fail) to keep budgets accurate.

## Files touched
- Code: `src/eesizer_core/sim/ngspice_runner.py`, `src/eesizer_core/strategies/patch_loop/strategy.py`
- Docs: `docs/specs/11_runtime_layout.md`

## Notes / Rationale
- Prevents Exec format errors from mis-arch vendor binaries and ensures integration tests skip cleanly.
- Keeps run artifacts auditable when a single iteration requires multiple attempts.
