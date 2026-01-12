# EEsizer Core Specifications (Contract-Level)

This folder contains the **contract-level specifications** for `eesizer_core`.

- `docs/wiki/` is for practical how-to guides.
- `docs/specs/` is the **source of truth** for interfaces, invariants, and rules.

If code and specs disagree, treat it as a bug:
- either update the implementation to match the spec, or
- revise the spec explicitly (include an ADR entry in `docs/design/`).

## Scope

These specs focus on the **EEsizer Core** building blocks:
- Artifacts (typed data objects)
- Operators (reusable tool wrappers / transforms)
- Policies (decision engines: LLM, RL, BO, heuristics)
- Strategies (workflow orchestrators)
- Provenance (hashes, logs, audit trail)

This split is designed to support future workloads (Task 1–4, RTL-Pilot) without turning the repo into a single giant LLM framework.

## Reading order

1. `00_index.md` (map of the spec)
2. `10_core_model.md` (the abstract model)
3. `20_artifact_contracts.md` (what data looks like)
4. `30_operator_contracts.md` (how tools are wrapped)
5. `40_patch_only_protocol.md` (LLM safety contract)
6. `90_step4_fixlist_and_acceptance.md` (what to fix and how to accept)

## Normative language

These specs use:
- **MUST / MUST NOT** for non-negotiable requirements
- **SHOULD / SHOULD NOT** for strong recommendations
- **MAY** for optional behaviors

## Links

- Architecture overview: `docs/wiki/03_architecture_contract.md`
- Patch-only guide: `docs/wiki/04_patch_only_protocol.md`
- Simulation and metrics guide: `docs/wiki/05_simulation_metrics.md`
- Dev workflow: `docs/wiki/02_dev_workflow.md`

