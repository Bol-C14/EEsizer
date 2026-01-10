
# agent.md — Unified Circuit Processor Dev Rules for LLMs 🤖

This file tells you, the LLM-based dev assistant, **how to work in this repo without turning it into spaghetti**.

You are not just writing code.  
You are maintaining a **research platform** that must stay:

- **Unified** (one abstraction, shared across tasks)
- **Auditable** (we can prove what happened)
- **Safe** (LLMs don’t silently rewrite circuits)
- **Extensible** (non-LLM ML / new tools can plug in later)
- **Documented** (docs/ is a first-class citizen, not an afterthought)

Treat everything below as **golden rules**.

---

## 0. Absolute boundaries

1. **Never modify `legacy/`**  
   - It is **read-only** reference code.
   - You may read it to understand behavior, but new work must go under `src/` and `docs/`.

2. **Never bypass the abstraction layers**  
   - Do not make “quick fixes” by wiring tools directly into strategies or policies.
   - Always route new behavior through the appropriate layer (Artifact / Domain / Operator / Policy / Strategy).

3. **Never write heavy logic into `contracts/`**  
   - `contracts/` defines **types and protocols**, not business logic.
   - No file I/O, no shell commands, no parsing, no tool invocation there.

4. **Never let an LLM generate full netlists as “ground truth”**  
   - LLMs can propose **parameter patches**, not rewrite entire circuits.
   - All structural modifications must go through **Patch + IR + Guards**.

---

## 1. Layering model (the mental map)

The codebase is organized as:

- `contracts/`  
  Pure definitions:
  - Artifacts (`CircuitSource`, `CircuitIR`, `ParamSpace`, `Patch`, `SimPlan`, `MetricSpec`, `RunResult`…)
  - Protocols (`Operator`, `Policy`, `Strategy`)
  - Infrastructure definitions (`Provenance`, `Errors`, `Enums`)

- `domain/`  
  Pure domain logic (no side effects):
  - SPICE parsing, tokenize, sanitation rules
  - signature computation
  - metric formulas (math only, no I/O)
  - small helper algorithms

- `operators/`  
  Tool-wrappers & stateful actions:
  - Netlist sanitization, parsing, indexing (wrapping `domain.spice.*`)
  - Simulation runners (ngspice, formal engines, etc.)
  - Metric extraction (wrapping domain formulas, file-based inputs)
  - Guards (topology invariants, constraint checks)

- `policies/`  
  Decision engines:
  - LLM-based policies
  - heuristic policies
  - RL / BO / other decision models
  - **They output `Patch` or structured plans; they do not run tools.**

- `strategies/`  
  Workflow orchestration:
  - Optimization loops
  - multi-step flows (map → optimize → verify → compare)
  - They call **operators** and **policies**, enforce **guards**, and decide when to stop.

- `runtime/`  
  Execution context:
  - `RunContext`, workspace paths, seeds, manifest writing
  - Registry of operators/policies if needed

- `docs/`  
  Human-facing and spec:
  - `wiki/` high-level explanations for humans
  - `specs/` precise contracts, JSON schemas, data format specs
  - `design/` architecture decision records (ADR)
  - `devlog/` periodic logs of changes
  - `reports/` experimental results, comparisons, draft deliverables

---

## 2. Dependency rules (you MUST respect these)

Think of dependencies as **one-way arrows**:

- `contracts` → nothing (only stdlib / typing)
- `domain` → `contracts` (types), stdlib
- `operators` → `contracts`, `domain`, `runtime`
- `policies` → `contracts` (and maybe `runtime` for logging), but **NOT** concrete operators
- `strategies` → `contracts`, injected operator/policy instances, `runtime`
- `docs` → describe all of the above; not imported into code

Golden rule:

> If you are about to write `import eesizer_core.X` from a lower layer into a higher layer, stop and re-evaluate. Lower layers must not depend on higher ones.

Examples of violations (do NOT do this):

- `contracts/artifacts.py` importing `domain.spice.parse`
- `domain/...` importing `runtime.context`
- `policies/...` importing `operators.ngspice_run` to run tools directly
- `operators/...` importing `strategies.sizing_loop`

---

## 3. Artifact / Operator / Policy / Strategy: what goes where

### 3.1 Artifacts

Artifacts live in `contracts.artifacts` and are the **only way** state is passed between layers.

Common ones:

- `CircuitSource`: raw netlist/HDL/schematic text + metadata
- `CircuitIR`: parsed, indexable representation (topology + param locations)
- `ParamSpace`: whitelist of tunable parameters, with bounds and frozen flags
- `Patch`: structured parameter updates (`PatchOp` list)
- `SimPlan`: which sims to run (dc/ac/tran/ams, with parameters)
- `MetricSpec` / `MetricValue` / `MetricsBundle`
- `RunResult`: final result, best candidate, history, stop reason

Rules:

- Artifacts can have **small helper methods** (`fingerprint()`, trivial `validate()`), but no side effects.
- Artifacts must be **serializable** and **hashable in content** (used in provenance).

### 3.2 Domain functions

Domain sets the **math and parsing rules**, without touching file paths or external tools.

Examples:

- `domain.spice.sanitize_text(raw_text) -> (clean_text, includes, warnings)`
- `domain.spice.parse_ir(clean_text) -> CircuitIR`
- `domain.spice.compute_topology_signature(circuit_ir) -> str`
- `domain.metrics.compute_gain(...) -> float`

Rules:

- No file I/O.
- No subprocess.
- No environment access.
- Only pure functions, given inputs, return outputs.

### 3.3 Operators

Operators adapt domain logic & tools to the runtime.

Pattern:

```python
class SomeOperator(Operator):
    name = "some_operator"
    version = "0.1"

    def run(self, inputs: Mapping[str, Any], ctx: RunContext) -> OperatorResult:
        # read artifacts from inputs
        # call domain functions OR external tools
        # produce new artifacts
        # record provenance
        ...
````

Example responsibilities:

* `SpiceSanitizeOperator`:

  * In: `CircuitSource`
  * Out: sanitized `CircuitSource` + includes + warnings
  * Uses: `domain.spice.sanitize_text`
* `SpiceIndexOperator`:

  * In: sanitized `CircuitSource`
  * Out: `CircuitIR` with `param_locs`
  * Uses: `domain.spice.parse_ir`
* `TopologySignatureOperator`:

  * In: `CircuitIR`
  * Out: `signature` string / artifact
  * Uses: `domain.spice.compute_topology_signature`

Rules:

* Must fill `OperatorResult.provenance` with input/output fingerprints.
* Must not embed long inline business rules that belong in `domain/` (e.g. fully custom parse logic).
* All file I/O (temporary decks, raw results) occurs here, not inside domain or contracts.

### 3.4 Policies

Policies are “brains that suggest what to do next”, but they **do not execute tools**.

They see an `Observation`:

* spec (objectives/constraints),
* current `CircuitSource` / `CircuitIR` / `ParamSpace`,
* current metrics,
* iteration index & history tail.

They output:

* `Patch` (parameter updates) or other structured decision (not free text netlist).

LLM policy example:

* Prompt: netlist (read-only), param table, metrics, objectives
* Output: JSON-only patch `{ "ops": [...], "stop": false, "notes": "..." }`
* The **system** (operators/strategy) applies the patch, not the policy itself.

Rules:

* Policies must not call ngspice, formal tools, etc.
* Policies must not write files or change state; they only return decisions.

### 3.5 Strategies

Strategies orchestrate the loop:

* Initialize IR, param space, sim plan, etc.
* For each iteration:

  * Compute metrics (via operators)
  * Build an `Observation`
  * Ask `Policy.propose(obs)`
  * Validate `Patch` (via guard operator)
  * Apply `Patch` (via operator)
  * Re-simulate & update history
* Decide stop reason (reached target / no improvement / budget / guard failures / policy stop)

Rules:

* Strategies don't know the tool internals.
* Strategies use only the stable interfaces from `contracts` + injected operator/policy instances.
* Stopping criteria must be explicit and reproducible, not hidden in LLM prompts.

---

## 4. Patch-based modification: non-negotiable safety rule

Whenever an LLM/policy wants to “change the circuit,” it must:

1. Propose a `Patch`:

   * Only `param_id`s from `ParamSpace`
   * `op ∈ {set, add, mul}`
   * values must be well-formed scalars (`float` or unit strings like `"180n"`)

2. `PatchValidateOperator`:

   * Rejects:

     * unknown param_ids
     * frozen parameters
     * values outside bounds
     * step too large (e.g. mul > 1.25)
   * Returns a structured result (ok/fail + reasons)

3. `PatchApplyOperator`:

   * Uses `CircuitIR.param_locs` to change **only the value part** in tokens.
   * Must not add/remove lines.
   * Must not change nodes or device types.

4. `TopologyInvariantGuard`:

   * Recomputes topology signature before/after apply.
   * If signature changed → veto and mark patch as invalid.

**You must NOT implement any path where a policy writes entire netlists directly.**

---

## 5. Docs discipline 📚

Whenever code changes affect behavior, interfaces, or architecture, you must also touch `docs/`.

### 5.1 Before coding

* **Read**:

  * `docs/wiki/01_unified_contract_overview.md`
  * relevant ADRs in `docs/design/ADR-*.md`
  * relevant specs in `docs/specs/` (if present)

### 5.2 When adding / changing functionality

You must consider updating **all适用**文档：

* `docs/wiki/`

  * If user-facing behavior or high-level flows changed.
* `docs/specs/`

  * If any contract, JSON schema, or data format changed.
* `docs/design/`

  * If you made a structural/architectural decision, add/update an ADR.
* `docs/devlog/`

  * For each substantial change (new feature / refactor), append a new devlog with:

    * Date & time
    * Scope
    * Files touched
    * Rationale & notes
    * Any migration notes or compatibility concerns
* `docs/reports/`

  * If you ran new experiments or produced comparative results relevant to deliverables.

**Rule of thumb:**
If a human collaborator reading only `docs/` would be surprised by the current code behavior, then the docs are outdated and must be updated.

---

## 6. How to handle a new task (standard workflow for you, the agent)

When the user asks you to add / modify something, follow this pipeline:

1. **Understand the level**

   * Is it:

     * a new Artifact type?
     * a new Operator (e.g. new sim tool)?
     * a new Policy (new strategy for decisions)?
     * a new Strategy (workflow)?
   * Or a refactor of IR / patch / guards?

2. **Update / create specs**

   * If you introduce new artifact fields / JSON formats / operator IO:

     * Update or create a spec under `docs/specs/`.
     * Make the format explicit (field names, types, allowed values, invariants).

3. **Implement in the right layer**

   * Put pure logic in `domain/`.
   * Wrap it in `operators/` to talk to files/tools and `RunContext`.
   * Connect via `Strategy` and `Policy` where needed.

4. **Add or update tests**

   * For domain: pure unit tests.
   * For operators: integration tests that mock external behavior if needed.
   * For strategies: smaller loops with mock policies/operators when possible.

5. **Update docs/devlog**

   * Create a new devlog entry in `docs/devlog/` summarizing:

     * what changed,
     * why,
     * how to use it,
     * any caveats.

6. **Preserve safety invariants**

   * Ensure patch-based modification & guards are still respected.
   * Never introduce hidden paths that bypass validation.

---

## 7. Code style & structure mini-rules

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

## 8. Summary

As the LLM dev agent, your job is not only to “make it work,” but to:

* Maintain the **unified contract** (Artifacts / Operators / Policies / Strategies / Guards).
* Keep **IR + Patch + Guard** as the backbone for safe circuit modification.
* Respect **dependency directions** and **layering**.
* Treat `docs/` as a first-class deliverable, updating it when the code changes.
* Avoid reinventing glue; instead, compose existing operators and policies.

If in doubt, **prefer more structure over less** and keep parsing/logic **pure and testable** in `domain/`, with thin, well-named Operators wrapping them.

Welcome to the Unified Circuit Processor. Don’t break the universe 🌌

