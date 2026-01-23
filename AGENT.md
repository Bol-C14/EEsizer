# AGENT.md — Development Rules for EEsizer (LLM + Humans)

This file tells you (human contributors and LLM-based dev assistants) how to work in this repo without breaking:

- **Safety** (policies cannot silently rewrite circuits)
- **Auditability** (we can prove what happened)
- **Reproducibility** (runs can be reconstructed)
- **Layering** (no “quick fix” cross-layer spaghetti)
- **Extensibility** (new tools/strategies can plug in cleanly)

Treat the rules below as **non-negotiable** unless an ADR explicitly changes them.

---

## 0. Absolute boundaries

1) **Never modify `legacy/`**  
   It is read-only reference code. New work goes under `src/` and `docs/`.

2) **Never bypass abstraction layers**  
   Do not wire tools directly into policies or domain logic. Use the correct layer.

3) **Never put business logic into `contracts/`**  
   `contracts/` defines types and protocols only.  
   No file I/O, no parsing, no subprocess, no network.

4) **Never let a policy generate a full netlist as ground truth**  
   Policies may propose **parameter-only patches** (or structured plans).  
   All circuit modification must go through **Patch + IR + Guards**.

---

## 1. Layering model (the mental map)

The codebase is organized as:

- `contracts/`  
  Pure definitions:
  - artifacts (`CircuitSource`, `CircuitIR`, `ParamSpace`, `Patch`, `SimPlan`, manifests, summaries…)
  - protocols (`Operator`, `Policy`, `Strategy`)
  - errors, enums, provenance types

- `domain/`  
  Pure domain logic (no side effects):
  - SPICE sanitation rules and parsing
  - tokenization and canonical forms
  - topology signature computation
  - metric formulas (math only)

- `operators/`  
  Tool wrappers and stateful actions:
  - netlist operators (sanitize / index / signature / patch apply)
  - simulation operators (deck build / ngspice run)
  - metrics operators (extract / compute)
  - guards (topology invariants, behavior constraints)
  - **Operators own side effects** and emit provenance.

- `policies/`  
  Decision engines (pure logic):
  - LLM policies, heuristics, or other models
  - output `Patch` or structured plans
  - **no tool execution, no file writes**

- `strategies/`  
  Workflow orchestration:
  - optimization loops (PatchLoop)
  - deterministic search (GridSearch)
  - robustness evaluation (CornerSearch)
  - multi-step orchestration (MultiAgentOrchestrator)
  - Strategies call operators and policies, enforce guards, and decide stop reasons.

- `runtime/`  
  Execution context and run I/O:
  - `RunContext`, recorder/writer, loaders
  - `ArtifactStore` and plan execution utilities
  - manifest writing and environment capture

- `docs/`  
  Human-facing documentation:
  - wiki, specs (schemas/formats), ADRs, devlogs, reports

---

## 2. Dependency rules (must respect)

Dependencies are one-way arrows:

- `contracts` → stdlib only
- `domain` → `contracts` (types) + stdlib
- `operators` → `contracts`, `domain`, `runtime`
- `policies` → `contracts` (and optionally lightweight runtime logging helpers), **not** concrete operators
- `strategies` → `contracts`, `runtime`, and **injected** operators/policies
- `docs` → describe, never imported into code

Golden rule:

> If you are about to write `import eesizer_core.X` from a lower layer into a higher layer, stop and re-evaluate. Lower layers must not depend on higher ones.

Examples of violations:

- `contracts/*` importing `domain.*`
- `domain/*` importing `runtime.*`
- `policies/*` importing simulation operators to run tools
- `operators/*` importing strategies

---

## 3. Unified attempt pipeline (shared contract)

All strategies that “evaluate a candidate” must follow the same pipeline and semantics.
If you find duplicate implementations, refactor toward a shared helper.

### 3.1 Canonical pipeline order

1) **Sanitize** the source (fail-closed include rules; `.control` removed)
2) **Index / parse** into `CircuitIR`
3) **Compute topology signature** (baseline structural fingerprint)
4) **Policy proposes a Patch**
5) **Patch validation guard** (param whitelist, frozen, bounds, step limits)
6) **Deterministic patch apply** (value-only token edits using `CircuitIR.param_locs`)
7) **Topology invariant guard** (signature must not change)
8) **Deck build + simulation** (ngspice or other tool operators)
9) **Metrics extraction / computation**
10) **Behavior guard(s)** (spec constraints)
11) **Record** results and provenance to the run directory

### 3.2 Failure semantics

- Guard failures must be recorded as structured `GuardCheck` results.
- Tool failures (simulation/metric errors) must:
  - be recorded,
  - contribute correct accounting (see budgets below),
  - and not corrupt the run directory structure.

---

## 4. Accounting, budgets, and reproducibility

### 4.1 Simulation run counting (non-negotiable)

If an evaluation runs multiple simulation kinds (e.g., AC + TRAN), **each kind counts**.

Rule:

> `sim_runs_total` must equal the number of times a simulator was invoked.

Do not “guess” counts at higher layers.
Counting belongs at the measurement layer where the simulator calls happen.

### 4.2 Stop reasons must be explicit

Strategies must use explicit stop reasons such as:

- target met
- no improvement
- budget exhausted (iterations, wall-time, sim runs)
- guard rejected
- policy requested stop

Stop logic must not be hidden inside prompts.

---

## 5. Run directory + manifest rules

Runs must be self-describing.

### 5.1 Required top-level files

A run directory should contain (names may vary slightly by strategy, but purpose must match):

- `manifest.json`  
  The index: config, environment, tool versions, and a file list.
- `summary.json`  
  Stop reason, best score, budgets used, key rollups.
- `history.jsonl`  
  One record per attempt/iteration with enough detail to audit decisions.

### 5.2 Optional strategy outputs (must be registered)

If present, these must be listed in `manifest.json`:

- `llm/` (prompts, responses, parsed patch JSON)
- `search/` (candidates, topk, pareto, corner sets, scoring summaries)
- `plans/` or equivalent (orchestrator plans and execution logs)
- `provenance/` operator call traces and fingerprints
- `report.md` or report artifacts

**Rule:** if a file is part of the run’s meaning, it must be discoverable from the manifest.

### 5.3 Path safety

All stage names and relative paths written to disk must be sanitized to prevent path traversal.

---

## 6. Patch-based modification: safety invariants

Whenever a policy wants to “change the circuit,” it must:

1) Propose a `Patch`
   - only `param_id`s from `ParamSpace`
   - `op ∈ {set, add, mul}`
   - values are well-formed scalars (float or engineering strings like `"180n"`)

2) Pass patch validation guards
   - reject unknown params, frozen params, out-of-bounds, unsafe steps

3) Apply patch deterministically
   - edit **only the value tokens**
   - do not add/remove lines
   - do not change nodes, element types, or models

4) Pass topology signature guard
   - signature before == signature after

There must be no alternate path that lets a policy rewrite netlists.

---

## 7. Corner evaluation rules (for robustness strategies)

Corner sets can easily become misleading if they fully overwrite design variables.

Guidelines:

- Prefer corners that represent **environment/variation parameters** (PVT, model corners, etc.).
- Avoid corners that **override all search parameters** unless explicitly requested.
- Baseline corner failures should be **recorded**, not used as a hard “gate” that prevents searching.
- Corner definitions and scoring rules must be written to `search/corner_set.json` (or equivalent) and listed in the manifest.

---

## 8. ArtifactStore and plan execution

If the repo uses `ArtifactStore` and `PlanExecutor`:

- Artifacts stored for audit must be serializable.
- If an artifact must be replayable across processes, it needs a stable schema and (ideally) a `type_tag` + decoder.
- Plan execution must record:
  - the plan document (JSON)
  - per-step outcomes (success/failure, outputs)
  - sub-run links (run IDs and paths)

---

## 9. Docs discipline

Whenever code changes affect behavior, interfaces, architecture, or run artifacts, update `docs/`.

At minimum consider:

- `README.md` (how to install/run; what exists now)
- `AGENT.md` (rules for layering, safety, run artifacts)
- `docs/wiki/` (high-level explanations)
- `docs/specs/` (schemas and formats)
- `docs/design/` (ADRs for structural decisions)
- `docs/devlog/` (short record of what changed and why)

Rule of thumb:

> If a collaborator reading only docs would be surprised by current behavior, docs are outdated.

---

## 10. Standard workflow checklist (for new work)

1) Identify the correct layer (artifact/domain/operator/policy/strategy/runtime).
2) Update or add specs if formats/contracts change.
3) Implement in the right layer with minimal cross-layer coupling.
4) Add tests:
   - domain: pure unit tests
   - operators: mock tools or integration markers
   - strategies: mock policies/operators where possible
5) Ensure run artifacts are complete and manifest-registered.
6) Update docs/devlog.
   * Create a new devlog entry in `docs/devlog/` summarizing:

     * what changed,
     * why,
     * how to use it,
     * any caveats.
7) **Preserve safety invariants**

   * Ensure patch-based modification & guards are still respected.
   * Never introduce hidden paths that bypass validation.
---

## 11. Style and structure

* One **Operator per file**, named after the action:
  `sanitize_spice.py`, `index_spice.py`, `ngspice_run.py`, `topology_invariant_guard.py`…
* No giant `utils.py`. If something feels like “just a utility,” think:

  * does it belong in `domain`? (pure logic)
  * or is it actually part of an Operator?
* Tests should mirror structure:

  * `tests/domain/test_spice_parse.py`
  * `tests/operators/test_spice_index_operator.py`
  * `tests/strategies/test_sizing_loop.py`

---

## 12. Summary

As the LLM dev agent, your job is not only to “make it work,” but to:

* Maintain the **unified contract** (Artifacts / Operators / Policies / Strategies / Guards).
* Keep **IR + Patch + Guard** as the backbone for safe circuit modification.
* Respect **dependency directions** and **layering**.
* Treat `docs/` as a first-class deliverable, updating it when the code changes.
* Avoid reinventing glue; instead, compose existing operators and policies.

If in doubt, **prefer more structure over less** and keep parsing/logic **pure and testable** in `domain/`, with thin, well-named Operators wrapping them.

Welcome to the Unified Circuit Processor. Don’t break the universe 🌌