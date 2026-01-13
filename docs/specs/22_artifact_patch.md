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

**Helpers**
- `ParamSpace.build(params)` builds the index (de-duping by `param_id`).
- `contains(param_id)` -> bool
- `get(param_id)` -> ParamDef | None

**Invariants**
- Param IDs MUST be unique.
- Frozen parameters MUST remain unchanged.

## 22.3 PatchOp

**Type:** `eesizer_core.contracts.artifacts.PatchOp`

**Fields**
- `param: str` (stable param id from ParamSpace)
- `op: PatchOpType` (`set`, `add`, `mul`)
- `value: Scalar` (number or string such as `"180n"`)
- `why: str` (optional rationale)

## 22.4 Patch

**Type:** `eesizer_core.contracts.artifacts.Patch`

**Fields**
- `ops: tuple[PatchOp, ...]`
- `stop: bool` (policy hint; optional)
- `notes: str` (free-form policy notes)

**Serialization note**
- The JSON schema uses top-level key `"patch"` for the ops list; inside the code/dataclass this is `ops`.

**Invariants**
- Patch MUST be safe to apply mechanically.
- Patch MUST NOT contain changes outside the ParamSpace.
- Multiplicative ops must use positive factors and are clamped by a max factor (default 10.0) during validation.

**Fingerprint**
- `Patch.fingerprint()` hashes the ops list (param/op/value/why) plus `stop`/`notes`.

## 22.5 Patch-only protocol (the ban on full-netlist rewriting)

Policies MUST output **Patch**, not full netlists.

Rationale:
- prevents topology drift
- prevents injection via `.control`, `.include`, etc.
- improves traceability (small diffs)

A JSON schema is provided at:
- `docs/specs/schemas/patch.schema.json`

Strategies SHOULD validate policy outputs against this schema.
