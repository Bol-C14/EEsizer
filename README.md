# EEsizer (Bol-C14/EEsizer)

EEsizer is a **modular, operator-driven framework** for analog / mixed-signal circuit processing and optimization.

It is designed for research teams and multi-tool workflows that need:

- **Safe, auditable netlist transformations**
- **Deterministic simulation + metrics pipelines**
- **Policy-driven optimization loops** (LLM, heuristics, and beyond)
- **Reproducible runs** with structured artifacts and manifests

**Key design principle**

> LLMs (and any policy) must **not** directly rewrite netlists.  
> Policies propose **parameter-only patches** (deltas), which are validated, applied, and guarded by deterministic local logic.

---

## Project status

This repository has been refactored from a legacy notebook-based codebase into a maintainable package.

### Milestones implemented so far

- ✅ **M1: Patch substrate + closed-loop patch optimization**
  - Parameter-only patching (`Patch`), validation, deterministic apply
  - Topology invariants and schema guards
  - Patch-loop strategy with heuristic and LLM patch policies

- ✅ **M2: Grid search strategy**
  - Deterministic coordinate / factorial sampling over `ParamSpace`
  - Search artifacts: candidates, top-k, Pareto sets

- ✅ **M3: Corner search strategy**
  - Evaluate candidates across a configurable corner set (global corners optional)
  - Corner set export, per-corner reporting, robust worst-corner scoring

- ✅ **M4: Multi-agent orchestration (experimental)**
  - Plan-based orchestration across strategies and tools
  - ArtifactStore and PlanExecutor to structure multi-step workflows

> The `legacy/` directory contains reference code and figures.  
> New development happens in `src/eesizer_core/`.

---

## Quick feature map

### Strategies (today)

- **PatchLoopStrategy**
  - Best for: iterative improvement under a single operating point
  - Typical policies: heuristic coordinate, LLM patch proposer

- **GridSearchStrategy**
  - Best for: deterministic exploration and baseline comparisons
  - Outputs: `search/candidates.json`, `search/topk.json`, `search/pareto.json`, `report.md`

- **CornerSearchStrategy**
  - Best for: robustness evaluation across corners (process/voltage/temp or param perturbations)
  - Defaults to per-parameter OAT corners; global corners are opt-in
  - Outputs: `search/corner_set.json`, `search/topk.json`, `search/pareto.json`, `report.md`

- **MultiAgentOrchestratorStrategy** (experimental)
  - Best for: multi-step plans (explore → refine → verify → compare)
  - Outputs: `orchestrator/plan.json` and `orchestrator/artifacts/index.json`

---

## Repository layout

```text
.
├── src/
│   └── eesizer_core/
│       ├── contracts/            # Stable “public” contracts (types + protocols)
│       ├── domain/               # Pure domain logic (no side effects)
│       ├── operators/            # Tool wrappers + stateful actions + guards
│       ├── policies/             # Decision logic that proposes patches/plans
│       ├── strategies/           # Orchestration loops (PatchLoop / Grid / Corner / Orchestrator)
│       ├── sim/                  # Deck builder + ngspice runner + source adapter
│       ├── metrics/              # Metric registry + metric computation operators
│       ├── runtime/              # RunContext, recorder, loaders, artifact store
│       └── analysis/             # Run comparisons, post-processing
├── tests/                        # Pytest suite (unit + mock + integration markers)
├── examples/                     # Runnable examples and demos
├── docs/                         # Wiki/specs/ADRs/devlogs/reports
├── legacy/                       # Legacy implementation (read-only reference)
├── pyproject.toml                # Packaging and dependencies
├── requirements*.lock            # Fully pinned environments (recommended)
└── README.md
```

---

## Core concepts (mental model)

### Artifacts (typed data)

Artifacts are typed containers that flow through the system:

- `CircuitSource`: raw netlist text + metadata
- `CircuitIR`: parsed, indexable representation (topology + token/param locations)
- `ParamSpace`: whitelist of tunable parameters with bounds and frozen flags
- `Patch`: parameter-only delta operations (`set/add/mul`)
- `SimPlan`: requested simulations (dc/ac/tran/…)
- `RunManifest` / `RunSummary`: run metadata, stop reason, file index, provenance links

Artifacts should be serializable and stable under hashing (used for provenance and caching).

### Operators (deterministic actions with provenance)

Operators are composable building blocks:

```text
Operator.run(inputs, ctx) -> OperatorResult(outputs, provenance)
```

Examples:

- sanitize / index / signature operators
- patch validate / apply operators
- simulation runners (ngspice)
- metrics extractors and compute operators
- guards (topology invariants, behavior constraints)

### Policies (proposers)

Policies propose “what to do next”:

```text
Policy.propose(observation, ctx) -> Patch  (or structured plan)
```

Policies must not run external tools directly and must not rewrite netlist text.

### Strategies (orchestrators)

Strategies coordinate operators and policies into a reproducible loop:

- build observation
- ask policy for a proposal
- validate + apply patch
- run simulations + compute metrics
- enforce guards and stop conditions
- record everything

---

## Installation

### Recommended: fully pinned environment (lock files)

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Choose the lock file that matches your needs:
pip install -r requirements-dev.lock   # dev + testing
# or:
pip install -r requirements.lock       # minimal runtime

pip install -e .
```

### Editable install with extras

```bash
pip install -e ".[dev]"
# Optional:
pip install -e ".[llm]"
pip install -e ".[viz]"
```

### Optional: ngspice

Some examples and integration tests require `ngspice` on your PATH:

```bash
ngspice -v
```

If ngspice is not installed, unit tests should still run; integration tests/examples will skip or fail gracefully.

---

## Running tests

```bash
PYTHONPATH=src pytest -q
```

To exclude integration tests:

```bash
PYTHONPATH=src pytest -q -m "not integration"
```

---

## Usage examples

Examples live in `examples/`. Typical scripts include:

- a single-run simulation + metrics example (AC/DC/TRAN)
- patch loop (heuristic / LLM policy)
- grid search
- corner search
- orchestrator (experimental)

Run scripts like:

```bash
PYTHONPATH=src python examples/<script>.py
```

> If you see `ModuleNotFoundError: eesizer_core`, install editable mode:  
> `pip install -e ".[dev]"`

---

## Run outputs (audit trail)

Each run writes a self-contained directory (usually under `examples/output/runs/<run_id>/` or a user-configured base dir).

A typical run directory contains:

```text
<run_id>/
  manifest.json              # the index: config, environment, file list, versions
  summary.json               # stop reason, best score, budget usage, etc.
  history.jsonl              # per-iteration events (attempts, guard results, metrics)
  best.json                   # best candidate patch + metrics summary
  report.md                   # human-readable report (strategy-dependent)
  llm/                         # prompts/responses (if an LLM policy/operator is used)
  search/                      # search artifacts (grid/corner strategies)
    candidates.json
    topk.json
    pareto.json
    corner_set.json
  provenance/                  # operator call traces and fingerprints
```

**Guiding rule**

> A run should be reproducible from: netlist text + patch JSON + tool versions + seed.

---

## Safety & correctness invariants

These are non-negotiable:

1) **Policies do not rewrite netlists**  
   Policies propose parameter patches only.

2) **Patch validation is fail-closed**  
   Unknown params, frozen params, out-of-bounds values, or unsafe steps are rejected.

3) **Topology invariants are enforced**  
   Topology signature must not change after patch apply.

4) **Sanitization precedes indexing/signature/simulation**  
   `.control` is removed; unsafe `.include` paths are rejected.

---

## How to extend EEsizer

### Add a new operator

1. Create a module under `src/eesizer_core/operators/...`
2. Implement the `Operator` contract:
   - validate inputs
   - call domain logic or external tools
   - return outputs + provenance
3. Add tests under `tests/`

### Add a new policy

- Implement a policy that outputs `Patch` (or a structured plan for orchestrators)
- Keep policies pure: no file writes, no tool calls
- Add tests with mocked observations

### Add a new strategy

- Compose existing operators/policies
- Reuse the shared attempt pipeline where possible
- Make stop reasons explicit and reproducible
- Ensure run artifacts are written and registered in `manifest.json`

---

## Troubleshooting

### `ModuleNotFoundError: eesizer_core`

Fix (choose one):

```bash
pip install -e .
pytest -q
```

or:

```bash
PYTHONPATH=src pytest -q
```

### `.include` is removed or warned

This is intentional. Include paths are restricted to avoid unsafe/irreproducible netlist execution.
Prefer controlled relative includes and avoid `..` and absolute paths.

### Signature changes unexpectedly

- check if you changed nodes / element names / models / parameter keys
- note: signature intentionally ignores numeric values but is sensitive to structure

---

## Contributing

- Keep `legacy/` read-only
- Put new functionality under `src/eesizer_core/`
- Every new operator/domain rule must have tests
- Prefer small PRs: one invariant / one operator / one test suite at a time
- Update `docs/` when behavior or formats change (wiki/specs/ADRs/devlog)

---

## Maintainers

- Repository: Bol-C14/EEsizer
