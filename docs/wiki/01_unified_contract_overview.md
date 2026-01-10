# Unified Contract Overview (UCP-style)

This document defines the highest-level contract for the project family (EEsizer, RTL-Pilot, and future tasks),
so that multiple researchers can develop different parts without creating glue/adapter chaos.

The core idea:
- Everything we handle is an **Artifact** (a structured representation of some design/state).
- Everything we do is an **Operator** (a function transforming artifacts).
- **Strategies** orchestrate operators to form workflows.
- **Policies** propose decisions (LLM / RL / BO / heuristics), but never execute tools directly.
- **Guards** are operators that enforce invariants and veto unsafe actions.

This keeps the system:
- unified (one coordinate system),
- testable (operators are unit-testable),
- explainable (provenance + decision records),
- extensible (new solvers/ML models become operators/policies without rewriting the framework).

---

## 0. Reading order

1. `docs/wiki/01_unified_contract_overview.md` (this file): what the system *is*
2. `docs/specs/contracts/*` : precise interfaces and data formats
3. `docs/design/ADR-*` : why the decisions were made
4. `docs/wiki/02_dev_workflow.md` : how to develop and contribute

---

## 1. The Universe model (mental model)

Let **U** be the set of all possible representations of a structure:
HDL code, SPICE netlists, schematic graphs, embeddings, screenshots, and so on.

We operate on U using a set of functions (operators):
- parse, transform, simulate, verify, report.

An "agent" (or a workflow) is simply a higher-order process that *composes* these functions to achieve a goal.

Key constraint:
- **We do not trust free-form generation of critical artifacts** (e.g., complete netlists).
- We instead constrain generation to **structured deltas** (patches), and apply them deterministically with guards.

---

## 2. Core building blocks and their responsibilities

### 2.1 Artifacts (data)
Artifacts are typed data objects with stable identities (hashable fingerprints).
Examples:

- **CircuitSource**: raw input (e.g., SPICE netlist text + metadata)
- **CircuitIR**: parsed intermediate representation (topology + parameter mapping)
- **ParamSpace**: a whitelist of tunable parameters + bounds + frozen flags
- **Patch**: a structured parameter delta proposed by a policy
- **SimPlan**: which simulations to run (ac/dc/tran/ams)
- **RunResult**: final best candidate + metrics + history
- **RunManifest/Provenance**: reproducibility records (inputs/outputs hashes, tool versions, commands)

> Rule: A Strategy/Operator may only communicate state via artifacts.
> No implicit global state, no hidden file paths as "API".

### 2.2 Operators (functions)
Operators are pure-ish transformations:

`outputs = Operator.run(inputs, ctx)`

Operators must:
- accept artifacts as inputs,
- produce artifacts as outputs,
- record provenance (inputs/outputs hashes, versions, commands),
- never silently mutate shared global state.

Operators can wrap real tools:
- ngspice, vivado, formal engines, parsers, metric extractors, etc.

### 2.3 Policies (decision proposals)
A Policy proposes actions but does NOT run tools.

`Patch = Policy.propose(Observation)`

Policies can be:
- LLM-based (tool calling optional, but output must remain structured),
- RL/NN models,
- Bayesian optimization,
- heuristics.

**A policy must not output a full netlist.**
It outputs structured deltas (Patch) or structured plans.

### 2.4 Strategies (workflow orchestration)
A Strategy composes operators and policies:

`RunResult = Strategy.run(spec, source, ctx)`

Strategies:
- control the loop,
- define stopping criteria,
- define which guards must pass,
- define when heavy checks (corners/formal) are triggered.

Strategies can be "fixed flows" or "agentic flows".
The difference is whether a Policy influences the next action,
but in all cases, execution is done via Operators.

### 2.5 Guards (safety / correctness gates)
Guards are operators that enforce invariants.
They can veto actions.

Minimum guard set (EEsizer):
- Topology invariant (signature unchanged)
- Parameter constraints (bounds/frozen/step size)
- Optional weak functional sanity checks (avoid absurd behaviors)

Digital tasks may add:
- Formal equivalence / property satisfaction
- COI coverage guards for regression selection

---

## 3. Why we ban "LLM outputs full netlist"

Generating full netlists is unsafe:
- topology can change silently (functional drift),
- injection risks (.control, .include, path traversal),
- debugging becomes impossible (diff is huge, no intent signal),
- reproducibility and auditing are weak.

Instead we use:
1. policy reads full context (netlist/spec/metrics)
2. policy outputs **Patch** = parameter deltas only
3. Patch is validated and applied deterministically
4. topology invariants are enforced by guards
5. the modified netlist is produced by a renderer, not by the LLM

This matches industrial practice:
- topology fixed, parameters tuned within constraints,
- checks and sign-off remain external and auditable.

---

## 4. Mapping to project tasks

This contract is shared across:
- Task 1/2: regression selection and testbench usefulness
  - Artifacts: RTLDesign, TestbenchSet, CoverageData, COISet, SelectionDecision
  - Operators: RunSim, ExtractCOI, CoverageMetrics, COI Guards
  - Policies: ranking/selection model
  - Strategy: select subset + guard coverage + report

- Task 3: security/adversarial testing
  - Artifacts: AttackSurfaceModel, AdversarialTest, VulnReport
  - Operators: knowledge retrieval, sim, violation detection, scoring
  - Policies: adversarial generation
  - Strategy: iterate -> detect -> report

- Task 4: technology node transfer (mixed-signal)
  - Artifacts: NodeSpec, MappingPlan, Analog/Digital variants
  - Operators: node mapping, EEsizer, mixed-signal sim, formal/regression reuse
  - Policies: mapping suggestions / tuning suggestions
  - Strategy: map -> optimize -> verify -> compare baseline

- RTL-Pilot:
  - Artifacts: Testbench, SimRunResult, FailureTrace, PatchSuggestion
  - Operators: RunSim, ExtractFailure, Coverage, Report
  - Policies: test generation/refinement
  - Strategy: loop test/refine/debug with guards

---

## 5. Contribution rules (high-level)

1. If you build a tool wrapper -> implement an Operator.
2. If you build a decision model -> implement a Policy.
3. If you build a workflow -> implement a Strategy.
4. All key state must be explicit artifacts.
5. All executions must record provenance.
6. Guards are mandatory for safety-critical transforms.
7. Contracts layer dependency rule: only standard library (incl. typing/dataclasses); keep third-party deps out to prevent toolchain bleed-through.
8. Internal imports must be explicit (e.g., `eesizer_core.contracts.CircuitSpec`, `eesizer_core.operators.netlist.TopologySignatureOperator`). The top-level `eesizer_core` wildcard exports are for demos/notebooks only—do not rely on them in core code to preserve boundaries.

---

## 6. Practical next steps

- Implement the Step1 contract skeleton (`docs/devlog/...`).
- Implement CircuitIR + signature (topology invariants).
- Implement Patch schema + validate/apply.
- Split simulation into operators (deck build / run / metrics).
- Rebuild EEsizer optimization loop on Patch-only policy outputs.
