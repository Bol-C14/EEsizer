# EEsizer (Structured Edition)

LLM-assisted analog/mixed-signal sizing with deterministic orchestration, ngspice in-loop simulation, and per-run isolation.

## What’s Inside
- **agent_test_gpt/** modular code:
  - `agent_gpt_openai.py` — entrypoint with `run_agent(user_question, user_netlist, run_dir=None)`.
  - `llm_client.py` — env-configurable models/temperature; structured logging.
  - `toolchain.py` — tool-call parsing/validation (allowlist + ordering).
  - `simulation_utils.py` — netlist munging, simulation builders, metrics (per-run output dirs).
  - `optimization.py` — LLM + simulation loop with retries/backoff.
  - `reporting.py` — guarded plotting and netlist copying.
  - `config.py` — path roots (project/package/resources/output).
  - `logging_utils.py` — structured logging + per-run log files.
  - `resources/ptm_*.txt` — model cards.
- **tests/**: coverage for prompts, tool parsing/validation, reporting guards, optimization runner (mocked), LLM client config.
- **docs/**: code review, change log.

## Requirements
- Python 3.12+
- ngspice installed and on PATH (or `NGSPICE_PATH` set) for real simulations.
- OpenAI-compatible LLM access (or mock).

## Quickstart
```bash
export OPENAI_API_KEY=...
export LLM_MODEL=gpt-4.1           # optional override
export LLM_FUNCTION_MODEL=gpt-4.1  # optional override
export LLM_TEMPERATURE=1           # optional override
export LOG_LEVEL=INFO              # DEBUG/INFO/WARNING/ERROR/CRITICAL
```

```python
from agent_test_gpt.agent_gpt_openai import run_agent

user_question = "Optimize this op-amp for >65dB gain, PM>60deg, power<5mW."
with open("initial_circuit_netlist/complementary_classAB_opamp.cir") as f:
    netlist = f.read()

results = run_agent(user_question, netlist)  # creates a per-run folder under output/runs/
print(results)
```

Outputs per run:
- `output/runs/<uuid>/`: netlist snapshots, metrics CSV, plots, logs (`logs/agent.log`), `run_manifest.json`.

## Testing
```bash
PYTHONPATH=. pytest
```

## Configuration Highlights
- Models/temperature via env: `LLM_MODEL`, `LLM_FUNCTION_MODEL`, `LLM_TEMPERATURE`.
- Logging: `LOG_LEVEL` and per-run file logging (`logs/agent.log`).
- Paths resolved relative to project root; PTM files under `agent_test_gpt/resources/`.

## Safety & Validation
- Tool-call allowlist and ordering checks.
- Netlist sanitization strips `.control`/unsafe includes; per-run output dirs prevent collisions.
- Simulation parsing asserts data presence; reporting validates CSVs/columns before plotting.
- LLM calls have configurable retries/backoff (`OptimizationConfig`).

## Roadmap
- Finer-grained simulation modules.
- Schema enforcement for LLM I/O.
- Artifact manifest expansion (prompts/responses) for reproducibility.
