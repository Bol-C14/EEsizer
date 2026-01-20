# 2026-01-20 18:30 — Dependency Locks and Numpy Floor

## Scope

- Add pip-tools lock files for base and extras.
- Lower numpy minimum version to 1.24 for broader compatibility.
- Document pinned install workflow.

## Files touched

- `pyproject.toml`
- `requirements.lock`
- `requirements-dev.lock`
- `requirements-llm.lock`
- `requirements-viz.lock`
- `requirements-opt.lock`
- `README.md`
- `docs/wiki/02_dev_workflow.md`

## Rationale & notes

- Lock files improve reproducibility for research runs and CI.
- Lower numpy floor avoids unnecessary install failures for environments pinned below 1.26.

## Regeneration

Use pip-tools:
```bash
pip-compile -o requirements.lock pyproject.toml
pip-compile --extra dev -o requirements-dev.lock pyproject.toml
pip-compile --extra llm -o requirements-llm.lock pyproject.toml
pip-compile --extra viz -o requirements-viz.lock pyproject.toml
pip-compile --extra opt -o requirements-opt.lock pyproject.toml
```
