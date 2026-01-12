# 22. Patch and Parameterization Artifacts

This doc specifies how EEsizer represents **what may change** and **how it changes**.

## 22.1 ParamDef

**Type:** `eesizer_core.contracts.artifacts.ParamDef`

**Fields**
- `param_id: str`  
  Globally unique within a circuit. Recommended form: `<elem_name>.<param_key>`.
- `unit: str`  
  Optional unit suffix (for example `"u"`, `"n"`, `"p"`), used by policies for readability.
- `lower: float | None` / `upper: float | None`  
  Optional bounds.
- `frozen: bool`  
  When true, policies and operators MUST not change it.
- `tags: tuple[str, ...]`

## 22.2 ParamSpace

**Type:** `eesizer_core.contracts.artifacts.ParamSpace`

**Fields**
- `params: tuple[ParamDef, ...]`

**Invariants**
- Param IDs MUST be unique.
- Frozen parameters MUST remain unchanged.

`ParamSpace.build()` MUST de-duplicate by `param_id` and raise on duplicates.

## 22.3 PatchOp

**Type:** `eesizer_core.contracts.artifacts.PatchOp`

**Fields**
- `param_id: str`
- `value: float`
- `mode: str = "set"`  
  Reserved for future modes. Today only `"set"` is supported.

## 22.4 Patch

**Type:** `eesizer_core.contracts.artifacts.Patch`

**Fields**
- `ops: tuple[PatchOp, ...]`
- `metadata: dict[str, Any] | None`  
  For policy explanations (reasoning summary, confidence, etc.).

**Invariants**
- Patch MUST be safe to apply mechanically.
- Patch MUST NOT contain changes outside the ParamSpace.

## 22.5 Patch-only protocol (the ban on full-netlist rewriting)

Policies MUST output **Patch**, not full netlists.

Rationale:
- prevents topology drift
- prevents injection via `.control`, `.include`, etc.
- improves traceability (small diffs)

A JSON schema is provided at:
- `docs/specs/schemas/patch.schema.json (mirrored from docs/templates/patch.schema.json)`

Strategies SHOULD validate policy outputs against this schema.

