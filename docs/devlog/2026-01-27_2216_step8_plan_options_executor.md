# 2026-01-27 2216 Step8 tool-calling Plan options + deterministic executor

- Added ToolCatalog export from `ToolRegistry`:
  - `src/eesizer_core/runtime/tool_registry.py` now stores tool metadata (schema/description/io/constraints)
  - `src/eesizer_core/runtime/tool_catalog.py` exports deterministic `tool_catalog.json` with stable sha256
- Added Step8 plan contracts + validation:
  - `src/eesizer_core/contracts/plan_options.py` (PlanOptions schema + strict validator)
  - `src/eesizer_core/runtime/plan_validator.py` (semantic checks: op allowlist, IO match, cfg/spec delta guardrails)
- Added Step8 session tool whitelist + execution plumbing:
  - `src/eesizer_core/runtime/session_plan_tools.py` (whitelist + IO contracts)
  - `src/eesizer_core/runtime/session_plan_execution.py` (concrete tool fns wired to `InteractiveSessionStrategy`)
  - `src/eesizer_core/runtime/session_plan_runner.py` (dry-run + execute with pause/resume via `requires_approval`)
  - `src/eesizer_core/runtime/plan_executor.py` now supports:
    - `requires_approval` pause points
    - resume via `start_idx`
    - passes `_action_outputs` into tool params for generic tools
- Added Step8 LLM plan generator operator:
  - `src/eesizer_core/operators/session_llm_plan_advice.py`
  - Context builder: `src/eesizer_core/operators/llm_context/build_planning_context.py`
  - Prompt template: `src/eesizer_core/operators/llm_prompt/build_plan_options_prompt.py`
- Session UX updates:
  - `src/eesizer_core/contracts/session.py` tracks `latest_plan_rev` + `plan_history`
  - `src/eesizer_core/runtime/session_store.py` adds `plan_dir()` + `mark_plan_decision()`
  - `src/eesizer_core/analysis/session_report.py` links latest plan artifacts in meta_report
- CLI: `tools/session/run_session.py`
  - New commands: `plan`, `show-plan`, `dry-run`, `execute-plan`
  - Supports `--mock-llm` and `--auto-approve` (pause/resume at `requires_approval`)
- Tests (no ngspice required):
  - `tests/test_step8_*`

Quick checks:
- `pytest -q tests/test_step8_*`
- End-to-end mock:
  - `PYTHONPATH=src python tools/session/run_session.py new --bench ota --mock --run-to-phase p1_grid`
  - `PYTHONPATH=src python tools/session/run_session.py plan --session-run-dir <run_dir> --provider mock --mock-llm`
  - `PYTHONPATH=src python tools/session/run_session.py show-plan --session-run-dir <run_dir>`
  - `PYTHONPATH=src python tools/session/run_session.py dry-run --session-run-dir <run_dir> --option-index 0`
  - `PYTHONPATH=src python tools/session/run_session.py execute-plan --session-run-dir <run_dir> --option-index 0 --mock --auto-approve`

