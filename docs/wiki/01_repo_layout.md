# Repository Layout

This repo is intentionally split into **legacy** and **refactor-track** code to prevent mixed paradigms.

## Top-level directories

### `legacy/` (read-only reference)
Contains the previous runnable implementation and historical run outputs.

- `legacy/legacy_eesizer/`: old package code (formerly `agent_test_gpt/*`)
- `legacy/tests/`: old tests (useful for regression reference)
- `legacy/docs/`: old docs and mapping notes
- `legacy/initial_circuit_netlist/`: example circuits
- `legacy/output/`: prior run artifacts (do not commit new outputs here)

**Rule:** do not add new features under `legacy/`.

### Refactor-track directories (new code)
Recommended new layout:

```
src/eesizer_core/          # reusable core: IR, constraints, patch engine, sim, metrics, strategies
src/eesizer_cli/           # thin CLI wrapper over eesizer_core
tests/                     # refactor-track tests (pytest)
examples/                  # minimal reproducible demos (used by CI)
docs/wiki/                 # this wiki
docs/templates/            # schemas / example configs
output/                    # local run artifacts (gitignored)
```

## Module ownership
- `eesizer_core/ir/*` owns: parsing, parameter-space extraction, topology signature.
- `eesizer_core/patch/*` owns: patch schema, validation, deterministic apply.
- `eesizer_core/sim/*` owns: deck building, simulator runners.
- `eesizer_core/metrics/*` owns: metric registry + implementations.
- `eesizer_core/strategies/*` owns: orchestration (loop, stopping rules, logging).
- `eesizer_core/policies/*` owns: decision-making (LLM, RL, BO); **no file I/O**, **no netlist editing**.

## Naming conventions
- Artifacts: `CircuitIR`, `ParamSpace`, `Patch`, `SimPlan`, `RawSimData`, `MetricSet`, `RunManifest`
- Operators end with `Operator` and implement `run()`
- Strategies end with `Strategy`
- Policies end with `Policy`
