# DevLog: Step4 Simulation Stack - Ngspice Runner Skeleton

- Date: 2026-01-11 09:00 (UTC)
- Scope: Kick off Step4 by introducing sim artifacts and an ngspice runner operator.

## Goals

1. Make ngspice execution an operator with explicit inputs/outputs.
2. Keep artifacts auditable: deck text + expected outputs + run metadata.
3. Provide a clear failure surface (missing binary, missing outputs) for reproducibility.

## Work Completed

- Added sim artifacts (`src/eesizer_core/sim/artifacts.py`):
  - `SpiceDeck` holds deck text, `SimKind`, and expected outputs (relative paths).
  - `RawSimData` records run directory, outputs, log path, command line, and stdout/stderr tails.
- Implemented `NgspiceRunOperator` (`src/eesizer_core/sim/ngspice_runner.py`):
  - Writes decks to `runs/<run_id>/<stage>/deck_<kind>.sp`, runs ngspice in batch mode, and captures logs.
  - Validates expected outputs from the deck and raises `SimulationError` with helpful tails on failure or missing binary.
  - Records provenance with deck hashes, expected outputs, command line, and output paths.
- Added tests (`tests/test_ngspice_runner.py`) using a mocked `subprocess.run` to cover success, missing binary, and missing expected outputs.

## Notes / Next Steps

- Deck builder and metrics operators are still pending; runner currently trusts the deck’s `expected_outputs`.
- When running tests locally without installing the package, set `PYTHONPATH=src` (devcontainer flow uses `pip install -e .[dev]`).
- Next PRs in Step4: deck builder operator, metric operators, and integration example using `RunContext` staging.
