# 41. Policy and Strategy Contracts

This spec defines the minimum structure required so different teams can swap in their own optimisers (LLM, BO, RL, heuristics) without rewriting the pipeline.

## 41.1 Policy contract

**Type:** `eesizer_core.contracts.policy.Policy`

A policy MAY have internal state, but it MUST expose:

- `name: str`
- `version: str`
- `propose(observation: Observation, ctx: Any) -> Patch`

### Observation structure (recommended)

Observation is a dataclass (`eesizer_core.contracts.policy.Observation`) and SHOULD include:
- `spec: CircuitSpec`
- `param_space: ParamSpace`
- `source: CircuitSource`
- `metrics: MetricsBundle`
- `iteration: int`
- `history_tail: list[dict[str, Any]]`
- `notes: dict[str, Any]`
  - Recommended: `notes["last_guard_failures"]` with recent guard failure reasons.
  - Recommended: `notes["current_score"]` and `notes["best_score"]` as scalar objective scores.
  - Recommended: `notes["param_values"]` mapping param_id -> numeric value for current state.
  - Recommended: `notes["last_guard_report"]` with structured guard feedback from the last attempt.
  - Recommended: `notes["attempt"]` retry index within the current iteration.

Policies SHOULD NOT require direct file access.

For LLM-backed policies, strategies MAY call helper methods (for example `build_request` and
`parse_response`) to keep tool invocation inside strategy/operator layers.

### Reference policies in repo
- `FixedSequencePolicy`: emits a fixed list of patches then returns `Patch(stop=True, notes="sequence_exhausted")`.
- `RandomNudgePolicy`: picks a non-frozen param and applies a small multiplicative nudge; stops if none available.
- `GreedyCoordinatePolicy`: coordinate descent / hill-climb with adaptive step size and guard-aware retries.

## 41.2 Strategy contract

**Type:** `eesizer_core.contracts.strategy.Strategy`

- `name: str`
- `version: str`
- `run(spec: CircuitSpec, source: CircuitSource, ctx: RunContext, cfg: StrategyConfig) -> RunResult`

### Strategy responsibilities

Strategies MUST:
- define stop conditions (max iters, max time, reached target)
- enforce Patch-only protocol
- ensure every tool call is wrapped by an Operator
- append a human-readable decision trace into `RunResult.history`

### Reference strategies in repo
- `PatchLoopStrategy`:
  - baseline: signature/IR -> ParamSpace -> grouped SimPlan per SimKind -> metrics -> objective eval
  - loop: policy -> PatchGuardOperator -> PatchApplyOperator -> TopologyGuardOperator -> sim run(s) -> BehaviorGuardOperator -> evaluate -> update best
  - stop reasons supported: `reached_target`, `policy_stop`, `no_improvement`, `max_iterations`, `budget_exhausted`, `guard_failed`
  - history records per-iteration: patch, guard report, attempts, signatures before/after, metrics, score, sim stage paths, warnings/errors
  - baseline sim errors are captured via guard reports and may stop early with `guard_failed`
- `GridSearchStrategy`:
  - deterministic coordinate/factorial sweep over `ParamSpace` (frozen params filtered by default)
  - explicit `param_ids` that overlap frozen params require `allow_param_ids_override_frozen=true`
  - outputs `search/candidates.json`, `search/ranges.json`, `search/candidates_meta.json`, `search/topk.json`,
    `search/pareto.json`, `report.md`
  - uses shared attempt pipeline for consistent sim/guard behavior and sim run accounting
- `CornerSearchStrategy`:
  - evaluates each candidate across a corner set and aggregates worst-case scores/losses
  - defaults to per-parameter OAT corners; `include_global_corners=false` unless explicitly enabled
  - corner overrides are relative by default (`corner_override_mode=add`) so candidate changes affect corners
  - baseline corner failures are recorded but do not gate search unless `require_baseline_corner_pass=true`
  - explicit `param_ids` that overlap frozen params require `allow_param_ids_override_frozen=true`
  - outputs `search/corner_set.json`, `search/topk.json`, `search/pareto.json`, `report.md`
- `MultiAgentOrchestratorStrategy`:
  - agent-driven plan selection that runs grid/corner search sub-strategies
  - records `orchestrator/plan.json`, `orchestrator/plan_execution.jsonl`, and artifact store index

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
