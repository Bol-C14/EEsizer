# 41. Policy and Strategy Contracts

This spec defines the minimum structure required so different teams can swap in their own optimisers (LLM, BO, RL, heuristics) without rewriting the pipeline.

## 41.1 Policy contract

**Type:** `eesizer_core.contracts.policy.Policy`

A policy MAY have internal state, but it MUST expose:

- `name: str`
- `version: str`
- `propose(observation: Mapping[str, Any]) -> Patch`

### Observation structure (recommended)

Observation SHOULD include:
- `spec: CircuitSpec`
- `param_space: ParamSpace`
- `last_patch: Patch | None`
- `metrics: MetricsBundle`
- `raw_data_refs: dict[str, str]` (paths under run dir)
- `constraints_status: dict[str, Any]`

Policies SHOULD NOT require direct file access.

## 41.2 Strategy contract

**Type:** `eesizer_core.contracts.strategy.Strategy`

- `name: str`
- `version: str`
- `run(source: CircuitSource, spec: CircuitSpec, ctx: RunContext, **cfg) -> RunResult`

### Strategy responsibilities

Strategies MUST:
- define stop conditions (max iters, max time, reached target)
- enforce Patch-only protocol
- ensure every tool call is wrapped by an Operator
- append a human-readable decision trace into `RunResult.history`

### Stop conditions (recommended set)

A sizing loop SHOULD stop when any of these are true:
- objectives satisfied within tolerance
- no improvement over N iterations
- step size below threshold
- budget exceeded (time, sims)

## 41.3 Formal and constraint checks

Strategies MAY compose additional operators:
- functional equivalence checkers
- property and assertion checkers
- design-rule validators

The architecture expects these checks to be operators so they are reusable across tasks.
