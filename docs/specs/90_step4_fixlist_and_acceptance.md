# 90. Step 4 Fix List and Acceptance Plan

This document is an executable checklist for Step 4 completion.

Step 4 focus (as agreed in the repo):
- contracts are explicit (Artifacts, Operators, Policy, Strategy)
- docs are sufficient for new contributors
- the patch-only safety protocol is enforceable
- the simulation + metrics stack is reusable and testable

## 90.1 What looks good already (current repo)

From the current `eesizer_core` implementation:
- ✅ Artifact dataclasses exist (`CircuitSource`, `CircuitSpec`, `ParamSpace`, `Patch`, ...)
- ✅ Operator scaffolding exists (DeckBuild, NgspiceRun, ComputeMetrics, PatchApply)
- ✅ Patch-only enforcement exists via `PatchApplyOperator` + topology signature
- ✅ Wiki docs exist in `docs/wiki/`
- ✅ Contracts are mirrored in `docs/specs/` (kept current)

Step 4 completion requires tightening a few gaps so the contract is unambiguous.

## 90.2 Fix list

### A. Documentation gaps (must fix)

1. **Add contract-level specs**
   - Status: ✅ provided in `docs/specs/` (this folder)
   - Action: keep these docs updated when interfaces change.

2. **Make run layout rules explicit**
   - Ensure everyone knows where logs, decks, csvs live.
   - Status: ✅ `11_runtime_layout.md`

3. **Define the Patch schema + validation usage**
   - Status: schema already exists at `docs/specs/schemas/patch.schema.json (mirrored from docs/templates/patch.schema.json)`.
   - Action: document how to validate in strategies.

### B. Code quality gaps (recommended)

1. **Typing cleanup in RunResult**
   - Status: ✅ `RunResult.history` now uses builtin `list[...]` (ruff clean).

2. **Stage name validation**
   - Status: ✅ ngspice runner validates `[A-Za-z0-9_-]+` before staging.

3. **Explicit tool version capture**
   - Status: ✅ ngspice runner records resolved path + best-effort version in provenance notes.

4. **Consistent error taxonomy**
   - Status: ⚠️ ongoing: operators mostly use ValidationError/SimulationError/MetricError; keep enforcing for new code.

### C. Testing gaps (must fix)

1. **Unit tests for PatchApplyOperator**
   - Status: ⚠️ partial; add explicit tests for invalid param, frozen, bounds, topology mismatch.

2. **Integration test for end-to-end loop skeleton (minimal)**
   - Status: ⚠️ pending; need a minimal strategy/example that builds deck -> runs ngspice -> computes metric -> returns RunResult.

3. **Golden fixture for parsing and patching**
   - Status: ⚠️ pending; add tiny SPICE fixture and expected patch diff.

### D. CI and developer ergonomics (strongly recommended)

1. Add a CI workflow that runs:
   - `ruff check`
   - `pytest -q` (unit only)
   - Status: ⚠️ pending

2. Add a Makefile or just document commands in `docs/wiki/02_dev_workflow.md`:
   - `make test`
   - `make lint`
   - Status: ⚠️ pending

## 90.3 Step-by-step execution plan

Execute in this order:

1. **Docs landing**
   - Ensure `docs/specs/00_index.md` links resolve.

2. **Fix typing + validation**
   - Apply the code fixes in section 90.2.B.

3. **Add tests**
   - Start with PatchApplyOperator unit tests.
   - Add minimal fixtures.

4. **Add the minimal end-to-end example**
   - Not a full optimiser, just a smoke test.

5. **Add CI**
   - Make it run on PRs and main.

## 90.4 Acceptance criteria

Step 4 is accepted when all of the following hold:

### Documentation acceptance
- `docs/specs/` exists and is coherent.
- Every operator in `eesizer_core` has a corresponding spec page.
- Patch-only protocol is explained and points to a schema.

### Code acceptance
- `python -m compileall src` succeeds.
- No operator writes outside the run dir.
- PatchApplyOperator rejects topology changes by default.

### Test acceptance
- Unit tests pass locally.
- Integration test (if enabled) passes on a machine with ngspice.

### Review checklist
A reviewer can answer "yes" to:
- Can I add a new metric without touching strategies?
- Can I add a new policy backend without touching operators?
- Can I add a new simulator runner without breaking metrics extraction?
- Can I reproduce a run given run_dir + manifest/logs?
