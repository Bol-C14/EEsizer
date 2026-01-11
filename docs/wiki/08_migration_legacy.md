# Migration Plan (Legacy -> Refactor Track)

Legacy code lives under `legacy/legacy_eesizer/` and remains the ground-truth reference until the new pipeline is stable.

## Migration principles
- Do not “wrap the monolith”.
- Extract reusable operators and recompose under the new contract.
- Keep the pipeline runnable at every step (small PRs).

## Suggested migration sequence
1) Create Artifacts + Operator interfaces (contracts only)
2) Implement CircuitIR indexing + topology signature
3) Implement Patch schema + validate/apply (patch-only)
4) Split simulation: deck builder / runner / metrics
5) Rebuild optimisation loop as a Strategy
6) Add LLM Policy that outputs Patch JSON only
7) Add manifests and reproducibility logs
8) Replace legacy runs with refactor-track runs in examples

## Legacy-to-new mapping (conceptual)
Legacy modules (reference):
- `legacy_eesizer/netlist_utils.py` -> `eesizer_core/ir/sanitize.py`
- `legacy_eesizer/simulation_utils.py` -> `eesizer_core/sim/*` + `eesizer_core/metrics/*`
- `legacy_eesizer/optimization.py` -> `eesizer_core/strategies/optimize_loop.py` (rewrite, do not copy)
- `legacy_eesizer/prompts.py` -> `eesizer_core/policies/llm_prompt_templates.py`
- `legacy_eesizer/toolchain.py` -> (later) plan builder / schema validator
- `legacy_eesizer/reporting.py` -> `eesizer_core/reporting/*` (post-processing only)

## Regression strategy
- Keep a tiny “golden” circuit and spec under `examples/`
- Compare total sim calls, best metrics, failure rate
