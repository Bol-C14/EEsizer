# 34. Operator: PatchApplyOperator

**Code:** `eesizer_core.operators.netlist.patch_apply.PatchApplyOperator`

## Purpose

Apply a `Patch` (parameter updates) to a circuit source or circuit IR.

This operator is the enforcement point for:
- patch-only optimisation
- topology invariance (default)

## Inputs

Required:
- `source: CircuitSource`
- `param_space: ParamSpace`
- `patch: Patch`
- Source netlists containing `.control/.endc` are rejected.

## Outputs

- `source: CircuitSource` (patched, same kind/name/metadata)
- `circuit_ir: CircuitIR` (normalized IR after patch)
- `topology_signature: str`

## Topology invariance

Topology guard is **always on**:
- the operator computes a signature of the original and patched circuit
- raises `ValidationError` if they differ (no opt-out)

The signature is defined in `eesizer_core.domain.spice.signature`.

## Determinism

Patch apply MUST be mechanical:
- only token-level replacements at `TokenLoc`s
- no heuristic regex on arbitrary lines unless strictly bounded and tested
- no edits to element names, node lists, `.include`, `.control`, or analysis directives.
- validation enforces bounds/frozen/unknown params and step constraints (mul factors must be positive and below the configured max).

## Provenance

Must include:
- original netlist fingerprint
- param space fingerprint (param ids)
- patch fingerprint
- topology signature before/after (and fingerprints of patched IR/netlist)
