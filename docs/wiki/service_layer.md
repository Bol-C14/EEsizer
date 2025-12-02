# Service Layer Overview

This page documents the modular services introduced in `eesizer_core/agents/services/`. Agents compose these services so each stage can be extended or tested independently.

## Services

- **PlannerService** (`planner.py`): Builds task decomposition and target-extraction prompts. Accepts goal, netlist summary, and targets; returns messages and a `NetlistSummary`.
- **ToolSelectionService** (`tool_selector.py`): Emits the tool call plan (default AC/DC/TRAN deck) and uses provider tool schemas to request function-calling responses.
- **SimulationService** (`simulation.py`): Applies structured param changes, writes a working copy of the netlist under the run’s simulation directory, executes the tool chain (or mock/real ngspice), aggregates metrics, and attaches simulation artifacts. It never overwrites the original input netlist.
- **OptimizationService** (`optimization.py`): Wraps `MetricOptimizer` plus reporting. Produces honest flags (`meets_gain`, `meets_power`, `targets_met`) without forcing metrics to hit targets, and writes summary/history/variant comparison artifacts.

## Safety defaults

- Netlist edits are applied to a run-scoped working copy (`metadata["working_netlist_path"]`); the source netlist stays untouched.
- Simulation failures retry up to a configurable limit, falling back to last-successful metrics instead of raising immediately.
- Optimization does not clamp metrics to the targets; it reports whether targets are met.
- Providers default to recorded fixtures; set `--live-llm` or credentials to enable live calls.

## Extending

- Add custom tool handlers via `ToolRegistry` and update `ToolSelectionService` plans to include them.
- Swap the scoring policy by passing a different `ScoringPolicy` or optimizer into `OptimizationService`.
- Provide additional prompt paths/overrides through `AgentConfig.prompt_paths` and `prompt_overrides`.

## CLI notes

`python -m pipeline.run ...` prints the JSON summary to stdout by default. Use `--quiet` to disable stdout emission if you only want files under `output/pipeline/<run-id>/`.
