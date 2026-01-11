# Testing & Quality Gates

Refactoring is only safe if tests are fast and meaningful.

## Test layers

### 1) Unit tests (fast)
- IR parsing/indexing
- topology signature invariants
- patch validation and deterministic apply
- metric calculators with synthetic data

### 2) Integration tests (medium)
- deck build + RawSimData fixtures
- end-to-end loop with mocked simulator
- schema validation for policy outputs

### 3) End-to-end (slow, optional in CI)
- real ngspice run on a tiny circuit (time-bounded)

## Required for every PR
- at least one new test for new logic
- no changes under `legacy/` unless explicitly migrating references

## Mocking ngspice
For CI stability:
- provide a `FakeSimulatorOperator` reading fixture outputs
- run strategies against it

## Suggested quality gates
- `pytest` must pass
- ruff/black recommended
- typed errors (no silent except)
