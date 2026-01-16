# 35. Operator: Guard Operators

Guard operators enforce invariants and return structured guard artifacts.

## 35.1 PatchGuardOperator

**Type:** `eesizer_core.operators.guards.PatchGuardOperator`

Inputs:
- `circuit_ir: CircuitIR`
- `param_space: ParamSpace`
- `patch: Patch`
- `spec: CircuitSpec` (optional)
- `guard_cfg: dict` (optional)

Outputs:
- `check: GuardCheck` (name: `"patch_guard"`, severity: `"hard"`)

Behavior:
- Validates patch against ParamSpace (unknown/frozen params, bounds, op types).
- Enforces optional limits:
  - `max_patch_ops`
  - `max_mul_factor`
  - `max_add_delta`
  - `wl_ratio_min`
- Enforces non-negative values for `*.w`, `*.l`, and passive `*.value`.
- Applies `CircuitSpec.constraints` for:
  - `kind="param_range"`: `{"param": "...", "lower": ..., "upper": ...}`
  - `kind="param_ratio_min"`: `{"lhs": "...", "rhs": "...", "min_ratio": ...}`
  - `kind="param_equal_group"`: `{"params": [...], "tol": ...}`

## 35.2 TopologyGuardOperator

**Type:** `eesizer_core.operators.guards.TopologyGuardOperator`

Inputs:
- `signature_before: str`
- `signature_after: str`

Outputs:
- `check: GuardCheck` (name: `"topology_guard"`, severity: `"hard"`)

Behavior:
- Fails if signatures differ, with a short diff summary in reasons.

## 35.3 BehaviorGuardOperator

**Type:** `eesizer_core.operators.guards.BehaviorGuardOperator`

Inputs:
- `metrics: MetricsBundle`
- `spec: CircuitSpec`
- `stage_map: dict[str, str]` (run_dir per SimKind)
- `guard_cfg: dict` (optional)

Outputs:
- `check: GuardCheck` (name: `"behavior_guard"`, severity `"hard"` or `"soft"`)

Behavior:
- Hard checks: objective metrics exist and are finite.
- Log scanning (configurable):
  - Uses `ngspice_{kind}.log` under each `stage_map[kind]`.
  - `log_hard_patterns` trigger hard failures.
  - `log_soft_patterns` trigger soft failures unless `soft_log_as_hard=True`.

## 35.4 GuardChainOperator

**Type:** `eesizer_core.operators.guards.GuardChainOperator`

Inputs:
- `checks: Iterable[GuardCheck]`

Outputs:
- `report: GuardReport`

Behavior:
- Aggregates checks and computes `ok`, `hard_fails`, `soft_fails`.

## 35.5 FormalGuardOperator (stub)

**Type:** `eesizer_core.operators.guards.FormalGuardOperator`

Outputs:
- `check: GuardCheck` with `ok=True`, `severity="soft"`, and `reasons=("not_implemented",)`
