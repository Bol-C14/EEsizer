# DevLog: Step3 Patch Domain Functions

- Date: 2026-01-10 18:00 (Europe/London)
- Scope: Add domain-only patch validation/application utilities with topology guard
- Repo: EEsizer

## Goals

1. Add domain-level patch validation against ParamSpace/CircuitIR.
2. Apply patches deterministically to CircuitIR lines without topology changes.
3. Add tests covering valid patches, rejects, and numeric ops.

## Work Completed

### Domain patching
- Implemented `validate_patch`, `apply_patch_to_ir`, and `apply_patch_with_topology_guard` in `src/eesizer_core/domain/spice/patching.py`.
- Added token-level update helper with strict numeric handling for add/mul.

### Tests
- Added `tests/test_step3_patch_domain.py` covering:
  - topology invariant on value changes,
  - unknown param rejection,
  - frozen param rejection,
  - numeric mul behavior.

## Notes / Limitations

- add/mul only support numeric values (unit parsing is intentionally deferred).
- CircuitIR elements/param_locs are reused; re-indexing is deferred until new params are supported.
