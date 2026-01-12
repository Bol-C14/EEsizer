# 30. Operator Contracts (General)

Operators are reusable building blocks that wrap tools or deterministic transforms.

## 30.1 Interface

**Type:** `eesizer_core.contracts.operators.Operator`

Operators MUST implement:

- `run(inputs: Mapping[str, Any], ctx: Any) -> OperatorResult`

Where:
- inputs are named artifacts or values
- ctx is a `RunContext` (for tool operators)

**Return:** `OperatorResult`
- `outputs: dict[str, Any]` (artifacts / paths / structured data)
- `provenance: Provenance | None`
- `logs: dict[str, str] | None`

## 30.2 Determinism

- Pure operators (no tools) SHOULD be deterministic given the same inputs.
- Tool operators MUST record enough provenance to explain non-determinism (versions, seeds, flags).

## 30.3 Side-effect discipline

- Operators MUST write only under `ctx.run_dir()`.
- Operators MUST NOT mutate input artifacts.

## 30.4 Error model

Operators MUST raise structured errors:
- `ValidationError` for invalid inputs or schema violations
- `SimulationError` for tool failures
- `MetricError` for metric extraction issues

Operators MUST NOT silently swallow errors.

## 30.5 Versioning

Every operator has:
- `name: str`
- `version: str`

Changing behavior in a way that affects outputs MUST bump `version`.
