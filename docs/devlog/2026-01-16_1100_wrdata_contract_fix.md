# 2026-01-16 11:00 — wrdata contract alignment

- Scope: align wrdata column contract with ngspice output format and stabilize metric extraction.
- Code: `src/eesizer_core/sim/deck_builder.py`, `src/eesizer_core/io/ngspice_wrdata.py`
- Tests: `tests/test_deck_builder.py`, `tests/test_ngspice_wrdata_loader.py`
- Docs: `docs/specs/31_operator_deck_build.md`, `docs/wiki/05_simulation_metrics.md`

Notes:
- Deck builder now emits `wrdata` vectors without explicitly listing scale columns; expected column order reflects ngspice’s repeated scale per vector.
- `load_wrdata_table` now honors `expected_columns` as the authoritative column names, even when headers are present.
