# Development Workflow

This doc describes how to develop, test, and run EEsizer in a reproducible way.

## Recommended environment: VSCode Dev Container
A devcontainer is provided under `.devcontainer/` (Ubuntu + Python + ngspice).

1. Open the repo in VSCode
2. “Reopen in Container”
3. In the container terminal:

```bash
python3 -m pip install -U pip
python3 -m pip install -e .[dev]
pytest -q
```

> If `pyproject.toml` is not yet present, add it as part of the refactor track (see [08_migration_legacy.md](08_migration_legacy.md)).

## Local environment (without container)
Requirements:
- Python 3.11+ (3.12 recommended)
- `ngspice` installed and available on PATH (or provide `NGSPICE_PATH`)

Suggested setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

## Environment variables
Typical variables for LLM-backed policies:
- `OPENAI_API_KEY`
- `LLM_MODEL` (e.g., `gpt-4.1` / `gpt-5`)
- `LLM_FUNCTION_MODEL` (optional)
- `LLM_TEMPERATURE` (default: 1)
- `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR)
- `NGSPICE_PATH` (if ngspice not on PATH)

## Running a minimal example
We maintain a “golden” example under `examples/`.

Target pattern:
```bash
python -m eesizer_cli run \
  --netlist examples/circuits/ota.cir \
  --spec examples/specs/ota.json \
  --out output/runs/
```

Expected outputs:
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
