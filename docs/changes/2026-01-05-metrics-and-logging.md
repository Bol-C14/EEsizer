# Change Document: Metrics Correctness & NGspice Logging (2025-12-31)

## Context
Addressed P0 correctness and observability gaps: AC metrics were using complex log10, unity bandwidth returned a span not a crossing, output swing used a hard-coded gain threshold with a sign error, and NGspice stdout/stderr was not captured in `agent.log`. Added a single source of truth for metric contracts and synthetic tests to lock behavior.

## What Changed

### 1) AC metrics are mathematically correct and fail-fast
- Introduced shared helpers `_load_ac_vector`, `_gain_db`, `_first_crossing` to enforce magnitude-based dB, column schema, and interpolated crossings.
- `bandwidth` now returns the first -3 dB crossing (Hz) from low-frequency gain (median of first k samples).
- `unity_bandwidth` now returns the 0 dB crossing frequency (UGBW), not a passband width.
- `ac_gain` reports low-frequency gain as the median of the first 5 samples (more stable than a single point).
- Missing/insufficient AC data now raises `MetricError` instead of emitting bogus values.

Files: `agent_test_gpt/simulation_utils.py`, `agent_test_gpt/metrics_contract.py`

### 2) Output swing semantics are explicit and configurable
- Added config/env knobs: `OUT_SWING_GAIN_TARGET` (default 10), `OUT_SWING_GAIN_TOL` (default 0.05).
- Output swing is computed via `_calc_output_swing`: find the linear region where slope ≥ (1 - tol) * gain_target and return `max(out) - min(out)` (positive swing).

Files: `agent_test_gpt/config.py`, `agent_test_gpt/simulation_utils.py`, `agent_test_gpt/metrics_contract.py`

### 3) Metric contracts documented and centralized
- New `agent_test_gpt/metrics_contract.py` defines `MetricSpec` and canonical `METRICS` entries (gain_db, bw_3db_hz, ugbw_hz, out_swing_v) with units, required files, and params.
- New `docs/metrics.md` documents definitions, units, and simulation conditions to align code/prompts/reporting.

### 4) NGspice stdout/stderr logging is controlled via env
- `run_ngspice` appends NGspice stdout/stderr to `output/runs/<uuid>/logs/agent.log` when `NGSPICE_LOG_TO_AGENT` is enabled (default on). Creates `logs/` if missing.

Files: `agent_test_gpt/config.py`, `agent_test_gpt/simulation_utils.py`, `README.md`

### 5) Synthetic algorithm tests added
- Pure-numpy tests lock -3 dB bandwidth, unity bandwidth, and output swing behavior without running ngspice.

Files: `tests/test_metrics_algorithms.py`

### 6) Toolchain/prompt alignment to per-sim NGspice (Option B)
- Prompt now states simulation_* auto-runs ngspice and run_ngspice must not be called; JSON-only tool_calls (no .control/path embedding).
- run_ngspice is no longer exposed to the LLM; simulation tools list returns empty.
- Added toolchain normalization: drops run_ngspice calls, dedupes simulations, and injects required simulations for analyses before validation.
- Added normalization tests covering run_ngspice removal, required-sim injection, and deduplication.

Files: `agent_test_gpt/prompts.py`, `agent_test_gpt/toolchain.py`, `agent_test_gpt/agent_gpt_openai.py`, `tests/test_toolchain_normalize.py`

## Behavior Changes / Notes
- AC metrics are always real-valued (magnitude-based) and will raise `MetricError` on empty/undersized data.
- UGBW now denotes the 0 dB crossing frequency (Hz); any prior interpretation as span is removed.
- Output swing is positive and tied to configurable gain target/tolerance; warnings emit when no linear region is found.
- NGspice logs now land in `logs/agent.log` per run by default; set `NGSPICE_LOG_TO_AGENT=0` to disable.
- Toolchains containing run_ngspice are cleaned before execution; analyses now get their prerequisite simulations injected if missing.

## How To Validate
- Targeted algorithms: `PYTHONPATH=. pytest tests/test_metrics_algorithms.py`
- Toolchain normalization: `PYTHONPATH=. pytest tests/test_toolchain_normalize.py`
- Full suite (if desired): `PYTHONPATH=. pytest`

## Follow-ups (Not Included)
- Wire `MetricSpec` into prompts/reporting so humans and LLMs consume the same metric definitions.
- Extend contracts/tests to phase margin, THD, CMRR, offset, ICMR, transient gain.
- Add structured handling of `MetricError` in optimization/toolchain to surface metric failures cleanly.
- Align prompt/toolchain messaging about per-simulation ngspice execution vs. single-call mode. 
