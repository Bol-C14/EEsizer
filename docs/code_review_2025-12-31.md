# Code Review – 2025-12-31 Snapshot

Scope: current `agent_test_gpt` code after recent refactors (per-run isolation, logging, validation). Focus is on (1) further structural/OOP refactors to improve reuse/maintainability, (2) remaining critical risks, (3) prep for the OOP plan hinted in `docs/eesizer_notebook_mapping.md`. No code changes made.

## Structural Refactor Opportunities
- **Central orchestration vs. monolith entrypoint** (`agent_test_gpt/agent_gpt_openai.py`): still mixes orchestration, prompt wiring, and runtime config in one function. Suggest extracting a thin `AgentRunner` class (builds contexts/deps/config, owns run_dir/log setup) and a `RunArtifacts` dataclass to carry outputs (netlist paths, csv, log).
- **Simulation utilities still jumbo** (`agent_test_gpt/simulation_utils.py`): contains builders, runners, and metrics. Split into modules or classes: `builders.py` (netlist augment), `runner.py` (ngspice I/O + timeouts), `metrics.py` (ac_gain/offset/etc.), `vgscheck.py` (filter/convert/format). Makes it easier to mock and to evolve per-sim execution.
- **Toolchain and validation cohesion** (`agent_test_gpt/toolchain.py`): parsing/validation are intertwined. Consider a `ToolChain` object with explicit states: parsed calls, validated sequence, inferred sim_types. This would allow future schema enforcement (JSON schema) and better error messages.
- **Config surface** (`agent_test_gpt/config.py`): still a bag of constants. Move to a typed settings object (e.g., `Settings`/`RunSettings`) that can merge env vars, defaults, and optional CLI flags. This will help OOP-ify runner wiring and reduce implicit globals.
- **Prompt builders** (`agent_test_gpt/prompts.py`): strings are hard-coded; aligning with a schema and versioning (e.g., `PromptSet` with ids) will ease future model swaps and notebook-mapping alignment.
- **Reporting** (`agent_test_gpt/reporting.py`): plotting is tightly coupled to CSV shape; consider a `History` object that loads, validates, and provides typed series to plotting functions. Add dependency injection for matplotlib to enable headless/testing.
- **Logging setup** (`agent_test_gpt/logging_utils.py`): currently configures root on first call. For cleaner embedding, provide an explicit `configure_logging(level, logfile)` helper and keep `get_logger` strictly accessor.

## Remaining Critical/High-Risk Items
- **Numeric correctness gaps (known, not yet mitigated)**: `cmrr_tran` still uses natural log vs. log10 and `stat_power` indexing assumptions (`agent_test_gpt/simulation_utils.py`). These can yield wrong metrics. Needs correction and tests.
- **LLM determinism and reproducibility**: manifest records model/temperature, but prompts/responses/tool chains are not persisted per run. Without snapshots, reruns are non-reproducible and debugging is hard. Consider saving prompt/response logs to the run directory.
- **Schema validation**: tool/task/target JSON from LLM is accepted after light parsing; no JSON schema enforcement. Malformed inputs could still sneak through. Adding explicit schema validation would reduce downstream surprises.
- **Dependency/version capture**: ngspice binary/version and resource hashes are not recorded. For defensible results, capture binary path/version and checksum PTM files into the manifest.

## OOP Prep Aligned to `eesizer_notebook_mapping.md`
- Define core domain objects:
  - `Netlist` (text + includes + resolved paths)
  - `ToolCall`/`ToolChain` (validated, typed)
  - `SimulationRun` (netlist variant + outputs + metrics)
  - `RunConfig`/`RunArtifacts`
  - `AgentRunner` (coordinates prompts → toolchain → sims → optimization)
- Introduce service interfaces:
  - `LLMClient` (with retries/backoff and model selection)
  - `Simulator` (ngspice runner with timeouts, artifact manifest)
  - `Reporter` (history loading + plotting)
- Plan module boundaries to match notebook mapping: prompts/schema, tool selection, netlist processing, simulation/metrics, optimization loop, reporting.
- Make dependency injection first-class (pass services/configs into runners) to simplify testing/mocking and multi-provider support.

## Testing and Coverage
- Added coverage for tool parsing/validation, reporting guards, LLM config, and a mocked optimization runner. Still missing:
  - Metric correctness tests (offset, cmrr_tran, stat_power, etc.).
  - End-to-end dry-run tests with mocked ngspice that assert artifacts (manifest, logs, CSV) are written.
  - Failure-path tests for schema violations and missing data files across all metrics.

## Suggested Next Steps
1) Fix numeric correctness in metrics (log10, column handling) and add unit tests.  
2) Persist prompts/responses/tool chains per run; extend manifest with ngspice version and resource hashes.  
3) Carve out simulation modules (builders/runner/metrics) and introduce `AgentRunner`/`RunSettings` classes.  
4) Add JSON schema validation for LLM inputs (tasks, targets, tool calls).  
5) Expand test suite with metric correctness and artifact-creation assertions.  
