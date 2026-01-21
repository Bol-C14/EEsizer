# 2026-01-21 15:37 — Milestone 1 Done (LLM Patch Policy)

## Scope

- Document Milestone 1 completion for the minimal LLM-in-loop patch flow.
- Point to the RC lowpass closed-loop demo and explain how to adapt it for OTA.
- Clarify guard feedback and LLM artifact logging expectations.

## Files touched

- `docs/policies/llm_patch.md`
- `docs/wiki/02_dev_workflow.md`
- `docs/wiki/04_patch_only_protocol.md`
- `README.md`

## Rationale & notes

- Milestone 1 centers on JSON-only patch proposals, prompt/response artifacts, and guard feedback loops.
- The RC lowpass demo uses `examples/run_patch_loop_llm.py` with `provider=mock` and the same loop works with
  `provider=openai` when credentials are available.
- OTA runs require a matching netlist/spec; the example script is the reference for wiring in the policy/strategy.

## Acceptance / verification

- Closed-loop RC demo is available via `examples/run_patch_loop_llm.py` (requires `ngspice`).
- OTA closed-loop flow uses the same strategy/policy pipeline with a provided netlist/spec (not in repo).
