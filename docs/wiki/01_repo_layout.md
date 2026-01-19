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
**Tooling rule:** ruff/mypy/pytest are not run on `legacy/`; it is reference-only and excluded from quality gates.

### Refactor-track directories (new code)
Current state:

```
src/eesizer_core/          # reusable core: contracts, domain.spice, operators.netlist, sim, metrics, runtime
src/eesizer_core/strategies/   # orchestration (patch loop, baselines)
src/eesizer_core/policies/     # decision-making (heuristics, LLM/RL/BO stubs)
tests/                     # refactor-track tests (pytest)
examples/                  # minimal reproducible demos (ngspice + metrics)
docs/wiki/                 # this wiki
docs/templates/            # schemas / example configs
output/                    # local run artifacts (gitignored)
```

Planned (not yet implemented):
```
src/eesizer_cli/           # thin CLI wrapper over eesizer_core (future)
```

## Module ownership
- `eesizer_core/contracts/*`: artifacts (CircuitIR/ParamSpace/Patch/SimPlan/MetricSpec...), enums, errors, operators/policy/strategy protocols, provenance.
- `eesizer_core/domain/spice/*`: netlist parsing/indexing, sanitize rules, topology signature, patch validation/apply logic (pure functions; no I/O).
- `eesizer_core/operators/netlist/*`: Operator wrappers for sanitize/index/signature/patch apply.
- `eesizer_core/sim/*`: deck builder, netlist bundle, ngspice runner (file/command execution).
- `eesizer_core/metrics/*`: MetricImplSpec, registry, compute functions (AC/DC/TRAN), compute operator.
- `eesizer_core/runtime/*`: RunContext (run_dir/workspace utilities).
- `eesizer_core/strategies/*`: workflow orchestration (loop, stopping rules, logging).
- `eesizer_core/policies/*`: decision-making (LLM, RL, BO); **no file I/O**, **no netlist editing**.
- (planned) `eesizer_cli/`: thin CLI that wires strategies/policies/operators.

## Naming conventions
- Artifacts: `CircuitIR`, `ParamSpace`, `Patch`, `SimPlan`, `RawSimData`, `MetricsBundle`, `RunManifest`
- Operators end with `Operator` and implement `run()`
- Strategies end with `Strategy`
- Policies end with `Policy`
