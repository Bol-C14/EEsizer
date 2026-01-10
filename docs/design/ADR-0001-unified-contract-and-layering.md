# ADR-0001: Unified Contract and Layering

- Status: Accepted
- Date: 2026-01-08
- Authors: Project team

## Context

We are developing multiple AI-assisted EDA workflows in parallel (EEsizer, RTL-Pilot, and Task 1–4 deliverables).
Historically, EEsizer was implemented as a large monolithic notebook, resulting in:
- severe coupling,
- weak reproducibility,
- unsafe netlist manipulation (LLM rewriting full netlists),
- high glue/adapter burden between subprojects and toolchains.

We need a unified abstraction that allows:
- independent development by multiple researchers,
- reuse of operators across tasks (simulation, formal checks, reporting),
- controlled integration of LLMs and non-LLM ML methods (RL/NN/embeddings),
- strong safety guards and auditability.

## Decision

We adopt a unified contract with four explicit layers:

1. **Artifacts**: typed representations of state and design (hashable and serializable).
2. **Operators**: deterministic transformations that consume/produce artifacts, record provenance, and wrap external tools.
3. **Policies**: decision engines that propose structured actions (Patch/Plan) but do not execute tools.
4. **Strategies**: workflow orchestrators that compose operators and policies, enforce guards, and define stopping criteria.

We additionally require:
- **Provenance** for every operator call and for each run (RunManifest),
- **Guards** as operators that validate constraints and veto unsafe transformations.

## Key Sub-Decisions

### D1: Ban "LLM outputs full netlist"
LLM output must not be used as a direct source of netlist truth.
Instead:
- LLM (as Policy) outputs structured `Patch` (parameter deltas only),
- Patch is validated against `ParamSpace` and constraints,
- Patch is applied deterministically on `CircuitIR`,
- Topology invariants are enforced via signature checks.

Rationale:
- prevents silent topology drift,
- prevents injection (e.g., .control, malicious includes),
- improves traceability and debugging (small diffs),
- aligns with industrial parameter tuning practices.

### D2: Keep "Policy" separate from "Strategy"
Policies (LLM/RL/BO) are replaceable modules.
Strategies define the workflow and stop conditions.
Rationale:
- avoids building a giant LLM-agent framework,
- enables later insertion of specialized ML models without refactoring core flow,
- improves testing (mock policies) and evaluation.

### D3: Operators must record provenance
Every tool invocation must record:
- input/output hashes,
- tool versions and commands,
- runtime information.

Rationale:
- enables comparative studies (Task4 deliverable iv),
- enables debugging and audit trails,
- supports reproducibility across machines.

## Consequences

### Positive
- Reduced glue code across the research group.
- Clear ownership boundaries: artifact vs operator vs strategy vs policy.
- Testability: operators can be unit-tested; strategies can be tested with mock policies.
- Extensibility: new solvers/ML models become policies/operators.
- Internal import rule: use explicit module paths (e.g., `eesizer_core.contracts.CircuitSpec`, `eesizer_core.operators.netlist.TopologySignatureOperator`). The top-level `eesizer_core` wildcard exports are for demos/notebooks; core code must not rely on them to preserve boundaries.

### Trade-offs
- Initial overhead: implementing IR/patch/provenance is upfront work.
- Requires discipline: contributors must adhere to contracts and not bypass guards.

## Implementation Plan (high-level)

1. Step1: implement contract skeleton (Artifacts/Operators/Policy/Strategy + provenance).
2. Step2: implement CircuitIR + signature.
3. Step3: implement Patch validate/apply + topology guard.
4. Step4: refactor simulation and metrics into operators.
5. Step5: rebuild EEsizer optimization loop with Patch-only policies.
