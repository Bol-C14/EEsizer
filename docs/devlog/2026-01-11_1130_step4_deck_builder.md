# DevLog: Step4 Deck Builder (AC)

- Date: 2026-01-11 11:30 (UTC)
- Scope: Add AC deck builder operator with deterministic .control injection, tests, and example.

## Goals

1. Generate ngspice-ready AC decks as an operator without mutating source netlists.
2. Standardize AC outputs to a CSV (`ac.csv`) with predictable column expressions.
3. Provide minimal example and tests to validate the new operator.

## Work Completed

- Implemented `DeckBuildOperator` (`src/eesizer_core/sim/deck_builder.py`):
  - Accepts raw netlist text and an AC-only `SimPlan`.
  - Injects a `.control` block with `set filetype=ascii`, `ac dec ...`, and `wrdata ac.csv frequency vdb(node) vp(node)` for requested nodes.
  - Produces a `SpiceDeck` with `expected_outputs={"ac_csv": "ac.csv"}` and deterministic deck text + provenance.
- Added tests (`tests/test_deck_builder.py`) to ensure control injection, AC command, wrdata shape, and validation for missing AC requests.
- Added example assets:
  - `examples/rc_lowpass.sp` (control-free netlist).
  - `examples/run_ac_once.py` to build the deck and run ngspice if available.

## Notes / Next Steps

- Currently AC-only; DC/TRAN deck builders remain to be added.
- Metric operators still need to parse `ac.csv` and support other analyses.
- Example run skips automatically when `ngspice` is absent.
