# EEsizer Wiki (Refactor Track)

EEsizer is being refactored from a notebook-derived workflow into a maintainable, testable package that supports **multi-agent / multi-policy optimisation** while keeping execution **deterministic, auditable, and safe**.

## Current repo state
- `legacy/` is a **frozen reference**: the previous runnable implementation, legacy docs, and past run artifacts.
- New development should happen **outside** `legacy/`, following the contracts and workflow in this wiki.
- Goal: make the core framework reusable beyond EEsizer (Task 1–4, RTL-Pilot), by standardising **Artifacts**, **Operators**, **Policies**, and **Strategies**.

## Quick links
- Dev workflow: [02_dev_workflow.md](02_dev_workflow.md)
- Architecture contract: [03_architecture_contract.md](03_architecture_contract.md)
- Patch-only protocol (no netlist rewriting by LLM): [04_patch_only_protocol.md](04_patch_only_protocol.md)
- Simulation & metrics: [05_simulation_metrics.md](05_simulation_metrics.md)
- Testing & quality: [06_testing_quality.md](06_testing_quality.md)
- Legacy migration plan: [08_migration_legacy.md](08_migration_legacy.md)

## Design rules (non-negotiable)
1. **LLM/Policy never outputs a full netlist.** It outputs a `Patch` (parameter deltas) only.
2. **Topology must be invariant** unless a dedicated “topology-edit” project explicitly opts in (not in scope for now).
3. Every tool invocation must be wrapped as an **Operator** with:
   - typed inputs/outputs
   - declared side-effects
   - provenance (version/hash/paths)
4. Every optimisation run must be reproducible from its `RunManifest`.

## Legacy references
- Notebook mapping → Artifacts/Operators/Strategy: `legacy/docs/eesizer_notebook_mapping.md`
- Code review snapshot: `legacy/docs/code_review_2025-12-31.md`
- Metric definitions: `legacy/docs/metrics.md`
