# Development Workflow

This doc describes how to develop, test, and run EEsizer in a reproducible way.

## Recommended environment: VSCode Dev Container
A devcontainer is provided under `.devcontainer/` (Ubuntu + Python + ngspice) and builds from the bundled `Dockerfile`.

1. Open the repo in VSCode
2. “Reopen in Container” (it will build the image from the Dockerfile)
3. The devcontainer runs `postCreateCommand` automatically (`python -m pip install -e ".[dev]"`). If it fails (network / resolver issues), rerun manually:

```bash
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
pytest -q
```

## Local environment (without container)
Requirements:
- Python 3.11+ (3.12 recommended)
- `ngspice` installed and available on PATH

Suggested setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

Pinned installs (recommended for reproducibility):
```bash
pip install -r requirements-dev.lock
pip install -e .
```

Extra lock files are available for optional features:
`requirements.lock`, `requirements-llm.lock`, `requirements-viz.lock`, `requirements-opt.lock`.

## Environment variables
Typical variables for LLM-backed policies:
- `OPENAI_API_KEY`
- `LLM_MODEL` (e.g., `gpt-4.1` / `gpt-5`)
- `LLM_FUNCTION_MODEL` (optional)
- `LLM_TEMPERATURE` (default: 1)
- `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR)
- ngspice must be available on PATH (advanced: pass `ngspice_bin=...` when constructing `NgspiceRunOperator`)

## Running a minimal example (current state)
Today the repo ships a runnable AC demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
PYTHONPATH=src python examples/run_ac_once.py
```

What it does:
- Builds an AC deck from `examples/rc_lowpass.sp`
- Runs ngspice via `NgspiceRunOperator` and writes results under `examples/output/runs/<run_id>/ac_example/`
- Computes default AC metrics (`ac_mag_db_at_1k`, `ac_unity_gain_freq`)

If ngspice is missing on PATH, the script will skip the run.

Outputs to expect:
- `runs/<run_id>/ac_example/deck_ac.sp`
- `runs/<run_id>/ac_example/ac.csv`
- `runs/<run_id>/ac_example/ngspice_ac.log`
- Printed metric values

## Future CLI target (not yet implemented)
Planned `eesizer_cli` flow (design target, not present in repo yet):
```bash
python -m eesizer_cli run \
  --netlist examples/circuits/ota.cir \
  --spec examples/specs/ota.json \
  --out output/runs/
```

Expected outputs once CLI exists:
- `output/runs/<run_id>/run_manifest.json`
- logs (`logs/agent.log` or `logs/run.log`)
- netlist snapshots (baseline + patched variants)
- metrics table (CSV/JSON)

## Debugging a failing run
1. Open the run manifest: `run_manifest.json`
2. Check simulator logs in the run folder
3. Re-run a single simulation step with the saved deck
4. If a metric is missing, confirm the required files exist (AC/TRAN/OP outputs)

## Git workflow (recommended)
- Keep PRs small and runnable.
- Every PR must include tests for new logic.
- Avoid committing anything under `output/`.

## “No surprises” rules
- No global mutable state
- No writing outside the run directory
- Every operator records provenance (inputs/outputs hash, tool version)
