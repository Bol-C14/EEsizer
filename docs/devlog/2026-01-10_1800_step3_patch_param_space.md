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
- Adjusted patch value replacement to recompute value span from the current token, preventing stale TokenLoc spans on length-changing updates.
- Added a minimal unit parser for f/p/n/u/m/k/meg/g/t suffixes to support add/mul with common SPICE values.
- Introduced shared `tokenize_spice_line` to keep parse and patch token indices identical.
- Reindexed CircuitIR after patch apply to avoid stale TokenLocs and added topology diff details when guard fails.
- Canonicalized param_ids to lowercase and made topology signature case-insensitive on element names.

### ParamSpace inference
- Added `ParamInferenceRules` (allow/deny regex, default units/bounds, optional WL ratio hint) and extended `infer_param_space_from_ir` accordingly.

### Operator patch apply
- Updated `PatchApplyOperator` in `src/eesizer_core/operators/netlist/patch_apply.py` to consume `CircuitSource` + `ParamSpace` + `Patch` and emit new `CircuitSource`, refreshed `CircuitIR`, and topology signature.
- Added `SpiceCanonicalizeOperator` to wrap sanitize+index+signature into a single canonicalizing step for strategies.

### Tests
- Added `tests/test_step3_patch_domain.py` covering:
  - topology invariant on value changes,
  - unknown param rejection,
  - frozen param rejection,
  - numeric mul behavior.
- Added a repeated-patch test to ensure length-changing updates remain correct.
- Added unit add/mul coverage for common suffix values.
- Added `tests/test_step3_patch_operator.py` for end-to-end operator validation.
- Added `tests/test_step3_param_space_from_ir.py` coverage for passive main values and .param parsing.
- Added continuation, .param set, subckt param, and bounds validation coverage.
- Adjusted tests to expect lowercase param_ids and deny/allow rule behavior.

### Provenance hashing
- Introduced `stable_hash_json` to avoid `str(obj)` hashing and updated provenance fingerprints (param_locs, includes, param_space) to use normalized JSON payloads.

### API hygiene
- Trimmed `eesizer_core/__init__.py` exports to contracts-only symbols to keep layer boundaries explicit (operators remain imported via explicit module paths).
- StrategyConfig reduced to budget/seed/notes to keep strategy-specific knobs out of the base contract.
- CircuitIR/Element made frozen with read-only mappings to prevent shared-state mutation.
- Enforced sanitize-before-index by rejecting `.control/.endc` in `SpiceIndexOperator`; shared line normalization for continuations prevents drift between sanitize/parse.

## Notes / Limitations

- add/mul only support numeric values (unit parsing is intentionally deferred).
- No subckt flattening or expression evaluation yet; patch ops assume direct token replacement on indexed lines.
