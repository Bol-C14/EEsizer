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
   - Status: ✅ covered by domain + operator tests.
   - How to verify: `pytest -q tests/test_step3_patch_domain.py tests/test_step3_patch_operator.py`

2. **Integration test for end-to-end loop skeleton (minimal)**
   - Status: ✅ provided via examples + integration test.
   - How to verify: `python examples/run_ac_once.py` (requires ngspice); `pytest -q -m integration`

3. **Golden fixture for parsing and patching**
   - Status: ✅ fixtures exist (examples netlists + tests CSV fixtures); inline netlists in patch tests cover parsing/apply.
   - How to verify: `pytest -q tests/test_step3_patch_domain.py tests/test_step3_patch_operator.py`

### D. CI and developer ergonomics (strongly recommended)

1. Add a CI workflow that runs:
   - `ruff check`
   - `pytest -q` (unit only)
   - Status: ✅ CI present at `.github/workflows/ci.yml`

2. Add a Makefile or just document commands in `docs/wiki/02_dev_workflow.md`:
   - `make test`
   - `make lint`
   - Status: ✅ commands documented in `docs/wiki/02_dev_workflow.md` (no Makefile required)

## 90.3 Step-by-step execution plan (current repo)

This section reflects the **current state** of the repo. Use it as a repeatable workflow when making changes.

1. **Verify the baseline locally**
   - Run the exact commands in **90.35 How to verify**.

2. **When you change any contract (Artifact/Operator)**
   - Update the corresponding doc under `docs/specs/`.
   - Add/adjust at least one unit test covering the new behaviour.

3. **When you change simulation outputs / wrdata columns**
   - Update `docs/specs/*` output contracts.
   - Update CSV fixtures under `tests/fixtures/` if needed.
   - Ensure metrics tests still pass.

4. **When you change dev workflow (devcontainer/CI/tool deps)**
   - Update `docs/wiki/02_dev_workflow.md`.
   - Ensure CI remains green.

5. **Optional improvements (track explicitly)**
   - Formatting policy (`ruff format`) and mypy enforcement decisions.

## 90.35 How to verify (exact commands)

Patch validation + topology guard:
- `pytest -q tests/test_step3_patch_domain.py`
- `pytest -q tests/test_step3_patch_operator.py`
- `pytest -q tests/test_step2_ir_signature.py`

Simulation runner + staging + provenance:
- `pytest -q tests/test_ngspice_runner.py`
- `pytest -q tests/test_ngspice_wrdata_loader.py`

Metrics (fixture-based):
- `pytest -q tests/test_metrics_algorithms.py`

End-to-end smoke (requires ngspice):
- `python examples/run_ac_once.py`
- `pytest -q -m integration`

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
