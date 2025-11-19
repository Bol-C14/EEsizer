# Agent Interface Blueprint

This document links the draft Python package under `eesizer_core/` with the notebook behaviors recorded in `agent_test_gpt/agent_gpt_openai_flow.md`. It specifies how each agent should plug into the shared interfaces, including I/O expectations and ordered call sequences.

## Core runtime summary

| Module | Purpose | Notes |
| --- | --- | --- |
| `eesizer_core.messaging` | Defines `Message`, `ToolCall`, and `MessageBundle` so notebook prompts map directly into structured envelopes. |
| `eesizer_core.context` | Provides `ExecutionContext` plus `ContextManager` to track working directories (`output/…`), message logs, and generated artifacts. |
| `eesizer_core.config` | Describes `SimulationConfig`, `OptimizationConfig`, and `AgentConfig` objects that capture ngspice paths, tolerance bands, and LLM model choices. |
| `eesizer_core.agents.base` | Supplies the abstract `Agent` lifecycle (plan → tool selection → simulation → optimization) mirroring the notebook loop. |

All downstream agents import these modules to ensure consistent data contracts.

## OpenAI sizing agent (`agent_gpt_openai.ipynb`)

**Inputs**
- Natural-language goal + PTM-90 CMOS netlist (`tasks_generation_question`, `netlist`).【F:agent_test_gpt/agent_gpt_openai_flow.md†L14-L39】
- Environment variables (`OPENAI_API_KEY`), configuration for ngspice binary paths, tolerance defaults.

**Outputs**
- Updated SPICE netlist copied to `output/90nm/netlist_cs_o3/` and CSV history + PDF plots.【F:agent_test_gpt/agent_gpt_openai_flow.md†L110-L168】

**Call order (pseudocode)**
```
with ContextManager(run_id) as ctx:
    agent = OpenAIAgent(config)
    plan_msgs = agent.build_plan(ctx)      # cells 5-15
    tool_msgs = agent.select_tools(ctx, plan_msgs)  # cells 18-24
    metrics = agent.run_simulation(ctx)    # executes tool_chain via ngspice
    results = agent.optimize(ctx, metrics) # cells 25-33
```
Planning prompts reuse `tasks_generation_template`, `target_value_SYSTEM_PROMPT`, node extraction, and classification cells. Tool selection wraps `simulation_function_explanation` plus function-call parsing. Simulation and optimization phases directly mirror `tool_calling` and `optimization` definitions described in the flow notes.【F:agent_test_gpt/agent_gpt_openai_flow.md†L41-L168】

## GPT-4o fast agent (`agent_4o.ipynb`)

**Inputs**
- Same goal/netlist payloads as the OpenAI baseline but tuned for the GPT-4o model variant.
- Config overrides: `model="gpt-4o"`, potentially shorter context windows and reduced simulation sets.

**Outputs**
- Rapid feasibility sizing report emphasizing first-pass metrics; expects shorter optimization history before handing control to higher-accuracy agents.

**Call order (pseudocode)**
```
agent = Gpt4oAgent(config.override(max_iterations=10))
plan = agent.build_plan(ctx)             # leaner prompts, drop redundant classification calls
tools = agent.select_tools(ctx, plan)    # prefer minimal AC + TRAN analyses
metrics = agent.run_simulation(ctx)      # single ngspice sweep
agent.optimize(ctx, metrics)             # coarse sizing, stops on first tolerance hit
```

## GPT-4o Mini agent (`agent_4omini.ipynb`)

**Inputs**
- Same circuit netlist but targeted at edge environments where API cost/latency is critical.
- Strict configuration caps on token budgets and runtime.

**Outputs**
- Lightweight hint set: recommended next actions for the primary agent plus optional parameter seeds.

**Call order (pseudocode)**
```
agent = Gpt4oMiniAgent(config.with_extra({"token_budget": 4_000}))
preflight = agent.build_plan(ctx)        # merges decomposition + tool choice into one round-trip
metrics = agent.run_simulation(ctx)      # reuses upstream results when available
agent.optimize(ctx, metrics)             # emits text-only guidance instead of editing netlists
```

## Claude 3.5 agent (`agent_claude3.5.ipynb`)

**Inputs**
- PTM/BSIM netlists stored in `agent_test_claude/` plus Anthropic API credentials.
- Configuration toggles for Claude-specific JSON/Markdown formatting quirks.

**Outputs**
- Updated rail-to-rail amplifier netlists and overlay plots stored next to the notebook artifacts.

**Call order (pseudocode)**
```
agent = Claude35Agent(config)
ctx = ContextManager(...)
plan_msgs = agent.build_plan(ctx)        # emphasises natural-language reasoning traces
tool_msgs = agent.select_tools(ctx, plan_msgs)
metrics = agent.run_simulation(ctx)      # identical ngspice driver, different prompt cadences
agent.optimize(ctx, metrics)             # multi-modal explanations for suggested edits
```

## Gemini 2.0 agent (`gemini_2.0.ipynb`)

**Inputs**
- Same transistor-level descriptions but routed through Google AI Studio credentials.
- Additional context about dataset privacy (Gemini requires explicit disclaimers).

**Outputs**
- Variation-aware sweeps plus CSV exports saved in `agent_test_gemini/`.

**Call order (pseudocode)**
```
agent = Gemini20Agent(config)
plan = agent.build_plan(ctx)             # ensures prompt conforms to Gemini function-call schema
tools = agent.select_tools(ctx, plan)
metrics = agent.run_simulation(ctx)      # may fan out to cloud-based circuit solvers
agent.optimize(ctx, metrics)             # integrates Monte Carlo/variation loops before finalize
```

## Next steps
- Replace notebook cells with concrete subclasses of `Agent` using the scaffolding provided here.
- Gradually migrate prompt templates into structured resources so each agent can version its behavior without editing Jupyter notebooks.
