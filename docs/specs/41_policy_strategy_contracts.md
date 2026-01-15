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
- `source: CircuitSource`
- `metrics: MetricsBundle`
- `iteration: int`
- `history_tail: list[dict[str, Any]]`
- `notes: dict[str, Any]`

Policies SHOULD NOT require direct file access.

### Reference policies in repo
- `FixedSequencePolicy`: emits a fixed list of patches then returns `Patch(stop=True, notes="sequence_exhausted")`.
- `RandomNudgePolicy`: picks a non-frozen param and applies a small multiplicative nudge; stops if none available.

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

### Reference strategy in repo
- `PatchLoopStrategy`:
  - baseline: signature/IR -> ParamSpace -> grouped SimPlan per SimKind -> metrics -> objective eval
  - loop: policy -> PatchApplyOperator (with topology guard) -> sim run(s) -> metrics -> evaluate -> update best
  - stop reasons supported: `reached_target`, `policy_stop`, `no_improvement`, `max_iterations`, `budget_exhausted`, `guard_failed`
  - history records per-iteration: patch, signatures before/after, metrics, score, sim stage paths, warnings/errors

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
