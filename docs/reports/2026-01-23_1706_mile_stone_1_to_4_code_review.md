# Full Audit – Issues & Fix Proposals (Post-Feature Development)

## P0 Issues (Functional correctness / algorithmic validity)

---

### **Issue P0-01: CornerSearch blocks search if baseline fails on any corner**

**Location**: `src/eesizer_core/strategies/corner_search.py` (baseline corner evaluation & early return)

**Symptom**
If the baseline design fails guard/simulation on **any** corner, the strategy immediately returns and never evaluates candidates.

**Impact**
Corner search is often used *because* the baseline is not robust.
Current behavior requires the baseline to already pass all corners, which makes CornerSearch unusable in many realistic analog scenarios.

**Root Cause**
Baseline corner evaluation is treated as a **hard gate** rather than a **reference measurement**.

**Fix Proposal (minimal, recommended)**

1. Treat baseline corner evaluation as informational only, not a gate.
2. When a baseline corner fails:

   * Record the failure (guard/sim error).
   * Assign baseline `worst_score = +∞` (or a configurable penalty).
   * **Continue into candidate evaluation**.
3. Only stop early when:

   * Budget is exhausted, or
   * An explicit config requires baseline corner pass.

**Optional Config**

```yaml
require_baseline_corner_pass: false   # default
```

**Required Tests**

* Baseline fails on one corner, candidate passes all corners → search continues and candidate can be selected.
* Ensure stop_reason is not `guard_failed`.

**Acceptance Criteria**

* CornerSearch proceeds even if baseline is not corner-robust.
* Baseline corner failures are visible in history but do not terminate search.

---

### **Issue P0-02: Global corners (`all_low/all_high`) fully override candidates**

**Location**: Corner set construction and corner patch application in `corner_search`

**Symptom**
`all_low` / `all_high` corners set **all parameters to bounds**, fully overwriting candidate patches.

**Impact**

* Metrics for these corners become **constant across all candidates**.
* Worst-corner score cannot improve regardless of search.
* Search direction becomes meaningless.

**Root Cause**
Corner definitions use absolute `set` on the same parameters being optimized.

**Fix Proposals (tiered, recommended order)**

**Option A – Minimal & Immediate (Recommended Default)**

1. Add config:

```yaml
include_global_corners: false   # default
```

2. Only include `all_low/all_high` when explicitly enabled.
3. Document that global corners are stress tests, not default optimization targets.

**Option B – Structural (Recommended Medium Term)**

1. Split parameters into:

   * `search_param_ids` (design variables)
   * `corner_param_ids` (corner disturbance variables, e.g. VDD, TEMP)
2. Global corners override **only `corner_param_ids`**, never search variables.

**Option C – Robust (Advanced)**

* Define corners as **relative perturbations** (`mul/add`) instead of absolute `set`.

**Required Tests**

* With default config, global corners are absent.
* With `include_global_corners=true`, they exist and are recorded.
* Worst-corner score improves across candidates when globals are disabled.

**Acceptance Criteria**

* By default, candidates can influence the worst corner.
* Global corners no longer lock optimization.

---

### **Issue P0-03: OAT corners can override the actively optimized parameter**

**Location**: Corner application logic + default `mode=coordinate`

**Symptom**

* Candidate modifies `param_i`.
* Corner `param_i_low/high` overwrites it.
* Candidate has no effect on that corner.

**Impact**

* Coordinate search becomes ineffective for robustness optimization.

**Root Cause**
Corner overrides and candidate patches operate on the same parameter set using absolute assignment.

**Fix Proposal**

1. In coordinate mode, exclude the currently optimized parameter from OAT corners **or**
2. Apply corner perturbations relative to candidate values **or**
3. Adopt `search_param_ids` vs `corner_param_ids` separation (recommended).

**Acceptance Criteria**

* A candidate modifying `param_i` can affect metrics at corners related to `param_i`.

---

## P1 Issues (Budget correctness, auditability, reproducibility)

---

### **Issue P1-01: `sim_runs` accounting is incorrect for multi-SimKind or partial failures**

**Location**

* `patch_loop/attempt.py`
* `grid_search.py`
* `corner_search.py`

**Symptom**

* When AC succeeds and TRAN fails, code often records `sim_runs = 1`.

**Impact**

* Budget enforcement (`max_sim_runs`) becomes unreliable.
* Run summaries are inaccurate (audit failure).

**Root Cause**
Simulation counting is done in exception handlers rather than where simulations actually execute.

**Fix Proposal (Recommended)**

1. Move `sim_runs` counting into `measure_metrics()`:

   * Increment per SimKind execution.
2. On failure:

   * Return a `MeasurementResult` with `runs_total / runs_ok / runs_failed`, **or**
   * Raise an exception carrying these counts.
3. Remove all hardcoded `sim_runs = 1` fallbacks.

**Required Tests**

* AC succeeds, TRAN fails → `runs_total == 2`, `runs_failed == 1`.
* Budget boundary test: stopping exactly at limit.

**Acceptance Criteria**

* Reported simulation counts match actual runner invocations.
* Budget stopping is deterministic and correct.

---

### **Issue P1-02: Search artifacts are not registered in `run_manifest.files`**

**Location**: `finalize_run()` / manifest assembly

**Symptom**
Files such as:

* `search/candidates.json`
* `search/topk.json`
* `search/pareto.json`
* `search/corner_set.json`

exist on disk but are missing from the manifest.

**Impact**

* Audit and reproduction are incomplete.
* Manifest does not reflect full run output.

**Root Cause**
Manifest registration only covers known inputs/LLM artifacts.

**Fix Proposal**

1. During finalization:

   * If `run_dir/search` exists, scan and register all `.json/.md` files.
2. Align behavior with `run_manifest.example.json`.

**Required Tests**

* Grid/corner search run → manifest includes all search artifacts.

**Acceptance Criteria**

* Manifest fully indexes all generated search files.

---

### **Issue P1-03: Budget stop conditions depend on incorrect `sim_runs`**

**Location**: All strategies using `max_sim_runs`

**Impact**

* Overshooting budgets or stopping late.

**Fix Proposal**

* Automatically resolved once **P1-01** is fixed.
* Add explicit boundary tests.

---

### **Issue P1-04: Frozen parameters are included in default `param_ids`**

**Location**: `grid_search.py`, `corner_search.py`

**Symptom**

* Candidates generated for frozen parameters, later rejected by guards.

**Impact**

* Wasted iterations, noisy history.

**Fix Proposal**

```python
param_ids = [p.param_id for p in param_space.params if not p.frozen]
```

* Respect user-provided `param_ids` explicitly.

**Acceptance Criteria**

* Frozen params excluded by default.

---

## P2 Issues (Maintainability, extensibility, long-term robustness)

---

### **Issue P2-01: Attempt pipeline logic duplicated across strategies**

**Location**

* `patch_loop/attempt.py`
* `grid_search.py`
* `corner_search.py`

**Impact**

* Bug fixes must be applied in multiple places.
* High regression risk.

**Fix Proposal**

* Reuse `run_attempt()` from patch_loop **or**
* Extract a shared `attempt_pipeline` module.

**Acceptance Criteria**

* All strategies use the same attempt execution path.

---

### **Issue P2-02: `corner_search.py` is excessively large (~1437 LOC)**

**Impact**

* Difficult to review, test, or reason about.

**Fix Proposal**
Refactor into a package:

```
strategies/corner_search/
  strategy.py
  corner_set.py
  measurement.py
  eval_candidate.py
  report.py
  config.py
  types.py
```

**Acceptance Criteria**

* Main strategy file ≤ ~300 LOC.
* Clear separation of responsibilities.

---

### **Issue P2-03: ArtifactStore cannot reliably support offline replay**

**Location**: `runtime/artifact_store.py`

**Symptom**

* In-memory artifacts are typed.
* Reloaded artifacts become raw dicts.

**Fix Options**

* A: Document ArtifactStore as audit-only (no replay guarantee).
* B: Add type tags + codecs to restore dataclasses.

**Acceptance Criteria**

* Behavior explicitly defined and documented.

---

### **Issue P2-04: Orchestrator duplicates run loading logic**

**Location**: Orchestrator vs `runtime/run_loader.py`

**Fix Proposal**

* Use `run_loader` universally.
* Remove custom loaders.

**Acceptance Criteria**

* Single source of truth for run parsing.

---

### **Issue P2-05: Packaging hygiene (generated files)**

**Risk**

* `__pycache__`, `.pyc`, `.egg-info`, `.DS_Store` leaking into distributions.

**Fix Proposal**

* Enforce via CI / pre-commit.
* Explicit packaging exclusions.

---

### **Issue P2-06: Documentation gaps for new strategies**

**Fix Proposal**

* README: Strategy overview.
* Per-strategy docs: inputs, outputs, stop conditions, artifacts.
* Orchestrator: minimal spec & dataflow.

---

## Suggested Execution Order (Practical)

1. **CornerSearch correctness**: P0-01 / P0-02 / P0-03
2. **Budget & accounting**: P1-01 / P1-03
3. **Audit completeness**: P1-02
4. **Refactor for DRY**: P2-01 / P2-02
5. **Runtime & replay clarity**: P2-03 / P2-04
6. **Docs & packaging**: P2-05 / P2-06