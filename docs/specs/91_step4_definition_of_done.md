# 91. Step 4 Definition of Done (DoD)

Scope: Step 4 delivers a reusable, testable, auditable **Simulation + Metrics stack** for EEsizer (and future tasks), using the **Artifacts / Operators** contract, and enforcing **Patch-only** safety boundaries.

## 0. Non-goals (Explicitly out of scope for Step 4)

- Full optimisation loop / multi-agent orchestration (Strategy layer) beyond minimal smoke tests
- Technology node transfer mapping flows (Task #4 deliverables beyond EEsizer core refactor)
- Formal equivalence of analogue function across patches (only guards + invariants)

## 1. Architecture & Separation of Concerns

### 1.1 Operators are cleanly separated

Required Operators exist and follow single-responsibility:
- DeckBuildOperator: `(CircuitSource or SpiceNetlist) + SimPlan -> SpiceDeck`
- NgspiceRunOperator: `SpiceDeck + RunContext -> RawSimData`
- ComputeMetricsOperator (or MetricOperator): `RawSimData + MetricSpec -> MetricsBundle`

Hard rules
- Deck building does not run ngspice and does not compute metrics.
- Runner does not compute metrics and does not mutate input netlists.
- Metrics calculators are pure (read outputs -> compute values), no hidden global paths.

Acceptance checks
- Code review confirms no cross-layer leakage (runner does not parse metrics, metrics do not spawn tools, etc.)
- Minimal dependency direction is respected (`metrics` can be used without `sim` running code, given fixtures)

## 2. Patch-only Safety Boundary (Step 4 must support it)

### 2.1 Patch-only is enforceable (not a convention)

- Optimisers/Policies never return a full netlist as output in the core runtime path.
- Patch apply is mechanical and constrained via allowlist.

Required components
- ParamSpace (authoritative allowlist)
- Patch schema (atomic ops: set/add/mul; includes `param`, `op`, `value`, optional `why`)
- PatchApplyOperator (or equivalent):
  - rejects unknown params
  - rejects frozen params
  - enforces bounds / step constraints
  - applies deltas deterministically
  - produces new netlist artifact

Acceptance checks
- Unit tests cover: unknown param, frozen param, bounds violation, illegal op/value types
- Patch apply never edits element names, node lists, `.include`, `.control`, or analysis directives

## 3. Topology & Guard Rails

### 3.1 Topology invariance (default-on)

- A topology signature exists and is computed from the circuit representation.
- Patch application must enforce: `topology_signature(before) == topology_signature(after)` by default.

Acceptance checks
- Unit test: patch changes numeric value -> signature unchanged
- Unit test: any attempted structural edit (if representable) -> rejected or signature mismatch detected

### 3.2 Netlist hardening rules

- Source netlist must not contain `.control` (or must be rejected/sanitised explicitly).
- Generated decks may inject `.control` for controlled output (wrdata).

Acceptance checks
- Test: input netlist with `.control` is rejected (fail closed)

## 4. Reproducibility & Provenance

### 4.1 Run directory discipline

All artefacts produced by Operators must live under: `<workspace_root>/runs/<run_id>/<stage>/...`

Hard rules
- No operator writes outside the run directory.
- Stage names must be sanitized (no path traversal).

Acceptance checks
- Unit test: stage name like `../evil` is rejected before any tool execution
- Manual/automated check: no outputs appear outside `<run_id>` directory

### 4.2 Provenance is recorded for tool runs

For each simulation run (RawSimData / provenance record), record at least:
- command line, cwd/workdir, returncode
- stdout/stderr tails + log path
- input deck hash (and source netlist hash if applicable)
- output file paths map
- ngspice resolved path + version (best-effort; failure must not crash run)

Acceptance checks
- RawSimData/provenance contains the fields above (or explicit “unknown” for version)
- If ngspice is missing, error message is actionable (points to install or env var)

## 5. Metrics Contract

### 5.1 Metric registry exists and prevents definition drift

- Metric definitions are registered centrally:
  - name, unit, required sim kind, required outputs, compute function

Hard rules
- Metrics must validate required outputs exist.
- Missing outputs must yield structured diagnostics (not silent NaN).

Acceptance checks
- Unit tests for at least 1–3 representative metrics (AC/DC/TRAN as available)
- Registry prevents duplicate/conflicting metric names without explicit override

### 5.2 Output format standardisation

- There is a documented and stable output contract for `wrdata` / CSV outputs:
  - filename(s)
  - required columns
  - units/scale conventions (Hz, s, V, A, dB, deg)

Acceptance checks
- A fixture CSV exists under `tests/fixtures/` and metrics compute from it deterministically

## 6. Documentation (Specs are “single source of truth”)

### 6.1 Specs must exist and match code

`docs/specs/` must include (at minimum):
- Artifact specs: `SimPlan`, `SpiceDeck`, `RawSimData`, `MetricsBundle`, `Patch`, `ParamSpace`
- Operator specs: `DeckBuildOperator`, `NgspiceRunOperator`, `ComputeMetricsOperator`, `PatchApplyOperator`
- Run layout/provenance spec

Hard rule
- No drift: Specs must reflect actual class names, input keys, and behaviour.

Acceptance checks
- `docs/specs/00_index.md` exists and links are not broken
- A reviewer can implement a new metric/operator using only specs + examples

### 6.2 Wiki consistency

- `docs/wiki/*` references must align with `docs/specs/*` and actual enums (SimKind, etc.)

Acceptance checks
- Manual doc review: no stale references like `op` if not supported

## 7. Quality Gates (Must pass locally)

### 7.1 Tests

Required
```
PYTHONPATH=src pytest -q
```
- All unit tests pass
- Integration tests may be skipped if ngspice unavailable, but must be runnable in CI/devcontainer

Optional but recommended
```
PYTHONPATH=src pytest -q -m integration
```

### 7.2 Lint / Style

Required
```
ruff check src tests
```

If formatting is enforced in this repo
```
ruff format --check src tests
```

### 7.3 Compile sanity

Required
```
python -m compileall src
```

Acceptance checks
- All commands above succeed with 0 errors on a clean environment

## 8. Repo Hygiene

- No tracked OS/IDE/cache junk (`.DS_Store`, `__pycache__`, `.pytest_cache`, `*.pyc`)
- `runs/` and `output/` are gitignored and never committed

Acceptance checks
- `git status` clean after running examples/tests

## 9. Minimal User Journey (Smoke Example)

A minimal runnable example must exist (even if not part of CI):
- build deck -> run ngspice -> compute ≥1 metric -> print/save MetricsBundle

Acceptance checks
- With ngspice installed:
```
python examples/run_ac_once.py
```
produces a run folder with:
- deck
- log
- output csv(s)
- metrics json/printout

## 10. Sign-off Checklist (One-page)

- [ ] Operator split is clean and reviewable
- [ ] Patch-only enforced via allowlist + deterministic apply
- [ ] Topology guard default-on
- [ ] Stage/path traversal prevented
- [ ] Provenance recorded (incl. ngspice version/path best-effort)
- [ ] Metrics registry + fixtures + unit tests exist
- [ ] docs/specs complete and not drifting
- [ ] `pytest` + `ruff` + `compileall` pass
- [ ] repo hygiene clean; no run artifacts committed
- [ ] minimal example runnable
