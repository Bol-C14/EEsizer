# Patch-Only Protocol (No Full-Netlist Editing)

This document defines the safety rule: **optimisers may only modify allowed parameters via a Patch.**

## Why
Allowing an LLM to emit a full netlist creates two risks:
1) accidental topology/function change
2) injection of unsafe directives (`.control`, unexpected `.include`, etc.)

Patch-only turns “generative editing” into “bounded search”.

## Patch definition
A Patch is a list of atomic operations:

- `param`: stable parameter id (from ParamSpace whitelist)
- `op`: one of `set`, `add`, `mul`
- `value`: number or a unit string (e.g., `180n`)
- `why`: optional rationale for explainability

Example:
```json
{
  "patch": [
    {"param": "M1.W", "op": "mul", "value": 1.1, "why": "increase gm"},
    {"param": "VBIAS.DC", "op": "add", "value": 0.02, "why": "fix headroom"}
  ],
  "stop": false
}
```

## ParamSpace definition
ParamSpace is the authoritative allowlist:
- which params exist
- current values
- bounds (min/max)
- frozen flags
- optional grouping (diff pair, current mirror, load)

Policies only see and reference ParamSpace. Unknown params are rejected.

## Validation rules (hard gate)
A patch must pass:
1) param exists in ParamSpace
2) not frozen
3) op is allowed for this param
4) within bounds after apply (or explicitly clamped if configured)
5) step size constraints (e.g., mul <= 1.25 per step)
6) cross-param constraints (ratio constraints) if specified

## Topology invariance rule (hard gate)
After applying a patch:
- compute topology signature again
- require `signature(new) == signature(old)`

Signature should be insensitive to numeric values and whitespace.

## Optional behavioural sanity checks (soft gate)
Low-cost checks to catch “function drift” even when topology is unchanged:
- OP converges (no fatal ngspice errors)
- basic sign of gain does not flip
- heuristic checks for extreme saturation/cutoff

These are guardrails, not formal equivalence proofs.

## LLM prompt rules (if using an LLM policy)
- Output must be Patch JSON only (enforced by schema validation).
- Never ask the LLM to print a modified netlist.

## LLM audit loop (recommended)
- Strategies should persist prompt/response artifacts under the run directory for auditability.
- Guard failures should be fed back to the policy via `Observation.notes` to avoid repeated invalid patches.
