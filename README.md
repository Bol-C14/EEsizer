# EEsizer (Bol-C14/EEsizer)

LLM-based AI Agent for Sizing of Analog and Mixed Signal Circuit

EEsizer is a **modular, operator-driven framework** for analog / mixed-signal circuit processing and optimization, designed to support:

* **Safe, auditable netlist transformations**
* **Deterministic simulation/metrics pipelines**
* **Policy-driven optimization loops** (LLM, RL, Bayesian optimization, heuristics)
* Future integration into a broader **Unified Circuit Processor** ecosystem (node transfer, verification copilot workflows, etc.)

> **Key design principle:**
> LLMs (and any “policy”) should **not** directly rewrite netlists.
> They should propose **parameter-only patches** (deltas), which are then applied by strict, deterministic, guard-railed local logic.

---

## Project Status

This repository has been refactored from a legacy notebook-based codebase into a maintainable package:

* ✅ **Step 1:** Contracts layer (Artifacts / Operators / Policy / Strategy / Errors / Provenance)
* ✅ **Step 2:** SPICE netlist canonicalization + lightweight IR + topology signature guard
* ✅ **Step 3:** Patch substrate (parameter-only editing with topology/schema guards)
* ✅ **Step 4:** Simulation stack (DeckBuildOperator + NgspiceRunOperator + ComputeMetricsOperator + metrics registry)
* ✅ **Step 5:** Strategy/Policy loops (PatchLoopStrategy + heuristic + LLM policies)

> The `legacy/` directory contains old code and references.
> New development happens in `src/eesizer_core/`.

---

## Why this architecture?

In research groups and multi-tool environments, the main pain points are:

* repeated “glue” adapters everywhere
* unclear boundaries between parsing, simulation, optimization, reporting
* code that works once but becomes impossible to extend safely (especially with LLM agents)

EEsizer solves this by enforcing:

* **A unified artifact model** (`CircuitSource`, `CircuitIR`, `ParamSpace`, `Patch`, etc.)
* **Composable operators** with consistent `run(inputs, ctx) -> outputs`
* **Strict guards** (topology signature + schema invariants)
* **Auditable provenance** (fingerprints, warnings, run metadata)

---

## Repository Layout

```
.
├── src/
│   └── eesizer_core/
│       ├── contracts/            # Stable “public” contracts (types + protocols)
│       │   ├── artifacts.py
│       │   ├── operators.py
│       │   ├── policy.py
│       │   ├── strategy.py
│       │   ├── errors.py
│       │   ├── enums.py
│       │   └── provenance.py
│       ├── domain/
│       │   └── spice/            # Pure functions for SPICE netlists (no side-effects)
│       │       ├── sanitize_rules.py
│       │       ├── parse.py
│       │       └── signature.py
│       ├── operators/
│       │   └── netlist/          # Operator wrappers: sanitize/index/signature/patch_apply
│       ├── sim/                  # Deck builder + ngspice runner + source adapter
│       ├── metrics/              # Metric registry + compute operators + algorithms
│       ├── io/                   # Shared IO helpers (e.g., wrdata loader)
│       └── runtime/              # Runtime context, run ids, shared execution utilities
├── tests/                        # Pytest suite (contracts + spice domain + operators + sim/metrics)
├── examples/                     # Minimal runnable examples / demos
├── legacy/                       # Legacy implementation (read-only reference)
├── pyproject.toml                # Packaging and dependencies
└── README.md
```

---

## Core Concepts (Quick Mental Model)

### Artifacts (Data)

Artifacts are *typed containers of information* that flow through the system.

Minimal set (already implemented):

* `CircuitSource`: raw netlist text + metadata
* `CircuitIR`: lightweight parsed representation (elements, nodes, token locations)
* `ParamSpace`: whitelist of controllable parameters
* `Patch`: parameter-only delta operations (set/add/mul)
* `SimPlan`: requested simulations (planned for later stages)
* `RunResult`: summary outputs + metrics + logs + provenance (later stages)

### Operators (Functions with contract)

Operators are composable building blocks:

```text
Operator.run(inputs, ctx) -> outputs
```

Operators should be:

* deterministic if possible
* auditable (provenance)
* side-effect-free unless explicitly stated (e.g., running ngspice later)

### Policy (Proposer)

Policy proposes “what to do next”:

* LLM policy
* RL policy
* Bayesian optimization policy
* heuristic policy

```text
Policy.propose(observation, ctx) -> Patch
```

### Strategy (Orchestrator)

Strategy runs a loop by composing operators and policies:

* canonicalize netlist
* apply patch
* simulate
* compute metrics
* stop conditions

```text
Strategy.run(spec, source, ctx, cfg) -> RunResult
```

---

## Getting Started

### 1) Requirements

* Python 3.10+ recommended (3.11 works well)
* (Optional, future) ngspice for simulation stage

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3) Install EEsizer in editable mode (with dev deps)

From repository root:

```bash
pip install -e ".[dev]"
```

For a fully pinned environment, use the lock files:

```bash
pip install -r requirements-dev.lock
pip install -e .
```

### 4) Run tests

```bash
PYTHONPATH=src pytest -q
```

> If you see `ModuleNotFoundError: eesizer_core`, it means the package is not installed in editable mode.
> Fix: `pip install -e ".[dev]"` and rerun `pytest`.

Example results (legacy reference):
![Optimization results for the opamp.](/legacy/figures/rail-to-rail-process.png)
![Variation results.](/legacy/figures/monte-carlo-10var.png)
![Performance comparison of different LLMs](/legacy/figures/performance-new.png)

---

## Dev Container (Recommended)

We support a VSCode Dev Container workflow to avoid environment drift across team members.

1. Install:

* VSCode
* “Dev Containers” extension
* Docker Desktop / Docker Engine

2. Open repository in VSCode
3. Run: **Dev Containers: Reopen in Container** (the image builds from `.devcontainer/Dockerfile`)

After container builds, dependencies will install automatically via `postCreateCommand`. If you need to rerun manually:

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest
```

---

## Usage: Canonicalize + Index + Signature (Step 2)

### What Step 2 provides

* **Sanitization**: remove `.control` blocks, restrict `.include` paths (fail-closed for dangerous includes)
* **Indexing**: build a lightweight IR that can locate editable tokens precisely
* **Topology Signature**: a stable fingerprint that changes when and only when **structure** changes

### Canonical netlist pipeline

Typical pipeline (operator layer):

1. `SpiceSanitizeOperator`
2. `IndexSpiceOperator`
3. `TopologySignatureOperator`

This ordering is enforced to prevent inconsistent IR/signature generation.

---

## Usage: Run AC sim + metrics (Step 4)

End-to-end demo (requires `ngspice` on PATH). If you installed the package in editable mode
(`pip install -e .`), you can omit `PYTHONPATH=src`:

```bash
PYTHONPATH=src python examples/run_ac_once.py
```

```bash
python examples/run_ac_once.py
```

This builds an AC deck for `examples/rc_lowpass.sp`, runs ngspice into `examples/output/runs/<run_id>/ac_example`, and computes AC metrics (`ac_mag_db_at_1k`, `ac_unity_gain_freq`). If ngspice is missing, the example will skip gracefully.

---

## Usage: Patch loop with LLMPatchPolicy (Milestone 1)

Minimal closed-loop demo (RC lowpass, mock provider by default):

```bash
PYTHONPATH=src python examples/run_patch_loop_llm.py --provider mock
```

Notes:
* Requires `ngspice` on PATH.
* For OpenAI, set `OPENAI_API_KEY` and use `--provider openai`.
* Run artifacts include `llm/` prompt/response files under `examples/output/runs/<run_id>/`.
* For other circuits, pass `--netlist` and update the spec in the script.

---

## Step 3 Recap: Patch Substrate (Parameter-only Editing)

### Motivation

Legacy EEsizer allowed the LLM to rewrite an entire netlist, which can accidentally:

* break topology
* change circuit function
* introduce illegal syntax or unwanted components

We instead enforce:

> The LLM returns only a `Patch` that says “which parameters to change and how”.
> The framework validates and applies it **deterministically** with strict constraints.

### Patch Contract (planned/being implemented)

A `Patch` is a list of operations:

* `param`: `"m1.w"` (must exist in `ParamSpace` and `CircuitIR.param_locs`)
* `op`: `"set" | "add" | "mul"`
* `value`: numeric literal (float/int) or engineering form `"180n"`, `"2u"`, `"1e-6"`
* `why`: optional explanation (recommended for audits)

**Hard rules**

* No netlist rewriting by policy
* No new elements
* No node changes
* Apply must preserve topology signature & schema

---

## Safety & Correctness Guards

We treat these as non-negotiable invariants:

1. **Topology Invariant Guard**

* signature before == signature after

2. **Schema Invariant Guard**

* set of `(element.param_keys)` and `param_locs` must not change

3. **Fail-Closed Input Rules**

* dangerous include paths rejected
* `.control` blocks removed
* indexing rejects non-sanitized input (to avoid parsing junk)

---

## Provenance & Reproducibility

The framework includes:

* `ArtifactFingerprint`: stable hash for artifact identity
* `Provenance`: operator-level trace of transformations
* `RunContext`: run id / seed / notes (foundation for experiment tracking)

Guiding rule:

> Any result should be reproducible from:
> netlist text + patch JSON + tool versions + seed.

---

## How to Extend the Framework

### Add a new Operator

1. Create a new module under `src/eesizer_core/operators/...`
2. Implement the `Operator` protocol:

   * validate inputs
   * call domain pure functions
   * return outputs + provenance
3. Add tests in `tests/`

### Add a new Domain Rule (SPICE)

Keep domain logic in `src/eesizer_core/domain/spice/`:

* no file IO
* no network
* deterministic if possible
* provide small focused functions

### Add a new Policy (LLM/RL/Heuristic)

Add to `src/eesizer_core/policies/` (or similar; not yet standardized):

* must output `Patch`
* must not modify netlist text
* keep prompts/LLM client isolated from domain logic

---

## Testing Guidelines

We use `pytest`.

Recommended test categories:

* `test_spice_sanitize.py`: sanitization correctness + include restrictions
* `test_spice_index.py`: parsing + param locations stable
* `test_ir_signature.py`: signature invariants
* (Step 3) `test_patch_validate.py`: constraints and fail-closed behavior
* (Step 3) `test_patch_apply.py`: deterministic patch application + guards

---

## Common Troubleshooting

### `ModuleNotFoundError: eesizer_core`

Fix (choose one):

```bash
pip install -e .
pytest
```

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python examples/run_ac_once.py
```

### `.include` is removed or warned

This is intentional. We restrict include paths to avoid unsafe/irreproducible netlist execution.
If you need includes, use controlled relative paths and avoid `..` and absolute paths.

### Signature changes unexpectedly

* check if you changed nodes / element names / models / param keys
* note: signature intentionally ignores numeric values but is sensitive to structure

---

## Roadmap (Short)

* Step 6: Optimization and deployment
  * richer metrics + corners
  * multi-agent orchestration (policy selection + tool calling)
  * integration toward node transfer workflows
  * performance: incremental IR updates for large netlists and long runs

---

## Contributing (Team Workflow)

* Keep `legacy/` read-only
* Add new functionality under `src/eesizer_core/`
* Every new operator/domain rule must have tests
* Internal code should import explicit modules (e.g., `eesizer_core.contracts`, `eesizer_core.operators.*`); top-level re-exports are for demos/notebooks
* Prefer small PRs:

  * one invariant / one operator / one test suite at a time

---

## License / Citation

* License: (add your license here)
* Citation: (optional, add paper/project citation)

---

## Contact / Maintainers

* Maintained by the Bol-C14 research workflow team.
* Repository: Bol-C14/EEsizer

---

## Acknowledgements

This work was made possible by Peter Denyer's PhD Scholarship at The University of Edinburgh.
