# 61. Extension Guide

This guide explains how to extend the framework without breaking contracts.

## 61.1 Add a new metric

1. Create a compute function:
   - signature: `(raw: RawSimData) -> MetricValue`
   - must validate `raw.kind` and required output files.

2. Register it in `MetricRegistry`:

```python
registry.register(MetricImplSpec(
  spec=MetricSpec(name="gain_db", description="...", unit="dB", sim_kind=SimKind.ac),
  compute=compute_gain_db,
))
```

3. Add tests:
   - unit test with a small fake CSV under `tests/fixtures/`

4. Document:
   - add to `docs/wiki/05_simulation_metrics.md` if user-facing

## 61.2 Add a new simulator operator

Example: Xyce or Spectre.

Requirements:
- must output the same `RawSimData` artifact
- must write into stage directories
- must record provenance (command, versions)

Steps:
1. Create `sim/xyce_runner.py` implementing an operator.
2. Keep the deck building separate (either reuse `SpiceDeck` or define a new deck artifact).
3. Add integration tests and mark them.

## 61.3 Add a new policy backend

LLM is only one backend. Others include:
- Bayesian optimisation (scipy)
- random search
- RL

Rules:
- policy returns Patch only
- policy never writes files directly

Steps:
1. Implement `Policy.propose()`.
2. Validate patch with schema + param space.
3. Add a simple demo strategy in `examples/`.

## 61.4 Add a new strategy

A strategy composes operators. Follow this skeleton:

- build deck
- run sim
- compute metrics
- decide patch
- apply patch
- loop

Checklist:
- stop conditions are explicit
- history is recorded
- failures are surfaced, not hidden
