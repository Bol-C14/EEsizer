# DevLog: Step1 Contract Skeleton

- Date: 2026-01-08 01:00 (Europe/London)
- Scope: Implement the Step1 contract skeleton for EEsizer refactor
- Repo: EEsizer (legacy moved under `/legacy`)

## Goals

1. Establish a unified, explicit contract to reduce coupling and glue adapters.
2. Separate responsibilities into Artifact / Operator / Policy / Strategy layers.
3. Prepare for future non-LLM methods (RL/NN/embeddings) by defining `Policy` as a replaceable decision component.
4. Lay foundation for reproducibility (provenance + run manifests).

## Key Decisions

- Adopt the unified contract (see `docs/wiki/01_unified_contract_overview.md`).
- Explicitly ban "LLM outputs full netlist" in the new architecture:
  - LLM (policy) must output structured patches (parameter deltas).
  - Deterministic patch application and strict guards will be implemented in later steps.

## Work Completed (Step1)

### Added new package skeleton
- Created `src/eesizer_core/` as the new core package location.
- Introduced `contracts/` module to host the canonical interfaces.

### Defined contracts
- Artifacts:
  - CircuitSource, CircuitSpec (objectives/constraints), ParamSpace, Patch
  - SimPlan, MetricSpec/MetricValue, RunResult
- Operators:
  - Standard `Operator.run(inputs, ctx) -> OperatorResult` interface
- Policy:
  - `Policy.propose(Observation) -> Patch`
- Strategy:
  - `Strategy.run(spec, source, ctx, cfg) -> RunResult`
- Provenance:
  - Hash-based fingerprints for artifact identity
  - RunManifest and per-operator provenance record structure

### Added runtime context
- Introduced `RunContext` with per-run workspace and run_id.
- This replaces legacy hard-coded output directory patterns over time.

### Added tests
- Added a smoke test that imports contracts and instantiates key dataclasses.

## Notes / Limitations

- No actual IR parsing, patch application, or simulation operators are implemented yet.
- Some fields remain intentionally abstract (e.g., Constraint payload) until Step2/3/4.
- The goal of Step1 is to freeze the "API surface" so implementation can proceed without structural churn.

## Next Steps

- Step2: implement CircuitIR (line-indexed IR) + signature computation.
- Step3: implement Patch validation + deterministic application + topology invariant guard.
- Step4: refactor ngspice and metric extraction into operators (deck/run/metrics).
- Step5: rebuild optimization loop as a Strategy using Policy-proposed Patches.

