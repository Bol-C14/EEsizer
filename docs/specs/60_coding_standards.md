# 60. Coding, Comments, and Documentation Standards

This spec defines contribution rules so the project remains maintainable as it grows into a multi-agent, multi-tool framework.

## 60.1 Repo hygiene

- New work MUST not be added to `/legacy`.
- `eesizer_core` is the reusable library.
- Anything experimental should live under `examples/` or a dedicated `experiments/` folder.
- Generated caches (`__pycache__`, `*.pyc`, `*.egg-info`, `.DS_Store`) MUST NOT be committed and MUST be excluded from packaging.

## 60.2 Style

- Python >= 3.10
- Formatter: ruff (line length 120)
- Type checker: mypy (optional but strongly recommended)

Every new public function or class MUST include:
- type hints
- a docstring describing purpose and invariants

## 60.3 Names

- Artifact classes are nouns: `CircuitSource`, `MetricsBundle`.
- Operator classes are verbs or `*Operator`: `PatchApplyOperator`, `ComputeMetricsOperator`.
- Policy and strategy names should match papers or baselines.

## 60.4 Logging and errors

- Do not `print` in library code.
- Use structured errors from `eesizer_core.contracts.errors`.

## 60.5 Docs rule

When adding or changing:
- an artifact: update `docs/specs/20_*`
- an operator: update `docs/specs/30_*`
- a protocol: update `docs/specs/40_*`

## 60.6 Tests

Minimum for any new operator:
- a unit test for validation (bad inputs fail)
- a unit test for a happy path using a minimal fixture

Tool-dependent tests must be marked `@pytest.mark.integration`.
