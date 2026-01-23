# Architecture Contract (Artifacts • Operators • Policies • Strategies)

This is the group-wide abstraction layer designed to reduce glue code and enable reuse across Task 1–4 and RTL-Pilot.

## 1) Core vocabulary

### Artifact
A typed data object representing “a structure”.
Examples: netlist text, parsed circuit graph, waveform files, metric vectors, coverage reports.

Artifacts must be:
- serializable (JSON/YAML where possible)
- hashable (stable fingerprint for provenance)
- immutable-by-default (new artifact per transformation)

### Operator
A reusable transformation: `Artifact -> Artifact` (or multiple in/out).
Operators may have side-effects (running tools) but must declare them and log provenance.

### Policy
A decision maker that proposes actions (e.g., a Patch).
Policies can be LLM-based, RL-based, Bayesian optimisation, or heuristics.

### Strategy
A workflow orchestrator that composes operators and policies into a pipeline with stopping rules.

## 2) Non-negotiable design rules
1. **Patch-only optimisation:** policies output patches, never full netlists.
2. **Topology invariance by default:** applying a patch must preserve topology signature.
3. **Deterministic application:** patch apply must be mechanical (no “best effort” regex unless strictly bounded and tested).
4. **Provenance everywhere:** every operator outputs enough info to reproduce a result.

## 3) Minimal core interfaces (conceptual)

### Operator interface
- `run(inputs: dict[str, Artifact], ctx: RunContext) -> dict[str, Artifact]`
- Returns a provenance record (hashes, versions, command lines, timestamps)
- Raises structured errors (no silent failure)

### Policy interface
- `propose(observation) -> Patch`
- Stateless or explicitly stateful (stored in RunManifest)
- Must not read/write files directly (except via ctx-managed logs)

### Strategy interface
- `run(source: CircuitSource, spec: CircuitSpec, ctx: RunContext) -> RunResult`

## 4) Where EEsizer fits
EEsizer becomes a set of Strategies (patch loop, grid search, corner search, multi-agent orchestrator), operating over shared Artifacts and Operators.

Future:
- Task 1–2 (verification): reuse the same Operator/Artifact discipline for coverage, COI extraction, formal checks.
- Task 4 (node transfer): add mapping operators and technology constraints, without rewriting runtime.
