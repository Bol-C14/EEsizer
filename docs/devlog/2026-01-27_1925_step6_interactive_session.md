# 2026-01-27 1925 Step6 interactive session + spec/cfg trace

- Added Step6 “interactive session” primitives:
  - `SessionStore` under `<run_dir>/session/` with deterministic JSON/JSONL writes:
    - `session_state.json`
    - `spec_trace.jsonl`, `cfg_trace.jsonl`
    - `spec_revs/spec_revNNNN.json`, `cfg_revs/cfg_revNNNN.json`
    - `checkpoints/*.json`
    - `meta_report.md`
- Added contracts for session/tracing/deltas/hashes:
  - `contracts/session.py`, `contracts/trace.py`, `contracts/deltas.py`, `contracts/hashes.py`
- Added pure delta application + diff operators:
  - `operators/apply_spec_delta.py`, `operators/apply_cfg_delta.py`, `operators/spec_diff.py`
- Added `InteractiveSessionStrategy`:
  - Phase pipeline: `p0_baseline` (NoOpt baseline) → `p1_grid` (GridSearch) → `p2_corner_validate` (CornerValidateOperator)
  - Checkpoint/continue semantics: if `input_hash` unchanged, phase is skipped and recorded as cached
  - Always regenerates `session/meta_report.md` after phase execution
- Added meta-report builder:
  - `analysis/session_report.py` creates a readable cross-phase report with links back to phase `report.md`
- Added minimal CLI:
  - `tools/session/run_session.py` supports `new`, `continue`, `inspect`
- Added Step6 unit tests (mocked; no ngspice required):
  - `tests/test_step6_*`

Quick checks:
- `pytest -q`
- `pytest -q -m integration`
- `PYTHONPATH=src python tools/session/run_session.py new --bench ota --run-to-phase p2_corner_validate --continue-after-baseline-pass`

