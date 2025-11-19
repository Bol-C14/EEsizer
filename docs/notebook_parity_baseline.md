# Notebook parity baseline

This repository keeps a single representative circuit netlist for regression checks that mirror the original `agent_gpt_openai.ipynb` notebook.

## Baseline circuit
- **Netlist:** `initial_circuit_netlist/ota.cir`
- **Models:** References `180nm_bulk.txt`, which is resolved from the repository `agent_test_gemini/` include path when ngspice runs.
- **Simulation modes:** The default control deck assembled by `SimpleSizingAgent` performs a DC sweep, an AC sweep, and a transient analysis to extract notebook-style measurements.

## Canonical metrics
The regression test compares the metrics emitted in `pipeline_result.json` for this circuit against golden values representing notebook behavior. Key metrics include:
- `gain_db`: AC gain in dB
- `power_mw`: Average power consumption in milliwatts
- `bandwidth_hz` and other derived quantities surfaced by the pipeline

These values live alongside the test under `tests/golden/notebook_parity_metrics.json` and can be adjusted if the reference notebook outputs change.
