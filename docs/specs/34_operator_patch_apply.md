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

## Outputs

- `circuit_source: CircuitSource` (same kind)
- `circuit_ir: CircuitIR` (normalized IR after patch)
- `topology_signature: str` (ToplogySignature.value)

## Topology invariance

Topology guard is **always on**:
- the operator computes a signature of the original and patched circuit
- raises `ValidationError` if they differ (no opt-out)

The signature is defined in `eesizer_core.domain.spice.signature`.

## Determinism

Patch apply MUST be mechanical:
- only token-level replacements at `TokenLoc`s
- no heuristic regex on arbitrary lines unless strictly bounded and tested

## Provenance

Must include:
- original netlist fingerprint
- patch fingerprint
- topology signature before/after
