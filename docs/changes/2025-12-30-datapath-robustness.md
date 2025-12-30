# Change Document: Data-Path Robustness (2025-12-30)

## Context
Recent runs surfaced brittle behavior in the `agent_test_gpt` pipeline where:
- LLM node responses were parsed too loosely, silently producing empty `output_nodes`/`source_names`.
- Simulation control blocks could be stacked into one netlist, making it easy for expected `.dat` artifacts (e.g. `output_ac.dat`) to be missing.
- Tool-chain validation rules were coupled to a single `run_ngspice` step, which no longer matches the preferred per-simulation execution (Option B).

This change set focuses on making the end-to-end datapath more deterministic and fail-fast, without introducing “silent fallback” behavior.

## What Changed

### 1) Node parsing is strict and schema-flexible
- `nodes_extract` now:
  - Accepts multiple key variants (singular/plural, mixed case) and both top-level dict and `{"nodes": [...]}` shapes.
  - Flattens list-valued fields and deduplicates while preserving order.
  - Raises errors when `output_nodes` or `source_names` are missing (no silent empty-list return).

Files:
- `agent_test_gpt/netlist_utils.py`
- `agent_test_gpt/agent_gpt_openai.py` (fails fast if node extraction returns empty outputs/sources)

### 2) Per-simulation NGspice execution (Option B)
- Instead of accumulating multiple `.control` blocks and running NGspice once, the runners now execute NGspice immediately per simulation type:
  - `dc_simulation` → runs NGspice → asserts `output_dc.dat`
  - `ac_simulation` → runs NGspice → asserts `output_ac.dat`
  - `transient_simulation` → runs NGspice → asserts `output_tran.dat`
- `run_ngspice` tool calls are treated as no-ops in the runners (logged), because NGspice is already executed by the simulation steps.
- After running simulations, Vgs/Vth reporting is derived from the most recent `op.txt` output.

Files:
- `agent_test_gpt/optimization.py`

### 3) Tool-chain validation aligned with per-simulation execution
- Validation no longer requires `run_ngspice` to appear before analysis tools.
- Validation enforces that each analysis tool appears after its prerequisite simulation step (e.g., `ac_gain` requires `ac_simulation`).

Files:
- `agent_test_gpt/toolchain.py`

### 4) LLM client configuration via env vars (and tests updated)
- LLM models and temperature are configurable via environment variables:
  - `LLM_MODEL`
  - `LLM_FUNCTION_MODEL`
  - `LLM_TEMPERATURE`

Files:
- `agent_test_gpt/llm_client.py`
- `tests/test_llm_client.py`

### 5) Code organization scaffold
- Created `agent_test_gpt/simulations/` package as the target home for future simulation orchestration refactors.

Files:
- `agent_test_gpt/simulations/__init__.py`

## Behavior Changes / Notes
- Runs will now fail earlier (and more clearly) if node extraction is incomplete.
- Tool chains that omit prerequisite simulations for analyses are rejected.
- Tool chains may omit `run_ngspice` entirely (as long as prerequisite simulations are present).

## How To Validate
- Unit tests: `PYTHONPATH=. /workspaces/EEsizer/.venv2/bin/python -m unittest discover -s tests -p 'test_*.py' -v`
- End-to-end run: `./.venv2/bin/python -m agent_test_gpt.agent_gpt_openai`

## Follow-ups (Not Included)
- Split `simulation_utils.py` into smaller modules under `agent_test_gpt/simulations/` (builders, runners, metrics).
- Improve prompt schema enforcement for nodes and tool-calls (JSON schema / stricter contracts).
- Add artifact manifest per run directory for reproducibility.
