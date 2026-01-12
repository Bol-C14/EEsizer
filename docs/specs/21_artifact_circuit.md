# 21. Circuit Artifacts

This spec matches the dataclasses in `eesizer_core.contracts.artifacts`.

## 21.1 CircuitSource

**Purpose:** the immutable input source for a circuit (netlist, HDL, etc.).

**Type:** `eesizer_core.contracts.artifacts.CircuitSource`

**Fields**
- `kind: SourceKind`
- `text: str`  
  Raw source text. For SPICE, this is the netlist content.
- `metadata: dict[str, Any] | None`  
  Optional context such as `base_dir` for `.include` resolution.

**Invariants**
- `kind` MUST be correct for the data in `text`.
- For SPICE netlists in EEsizer, `kind` MUST be `SourceKind.spice_netlist`.

**Fingerprint**
- `CircuitSource.fingerprint()` MUST hash:
  - `kind` value
  - `text`
  - `metadata` (stable JSON)

## 21.2 CircuitSpec

**Purpose:** the declared intent for optimisation (goals + constraints).

**Type:** `eesizer_core.contracts.artifacts.CircuitSpec`

**Fields**
- `objectives: tuple[Objective, ...]`
- `constraints: tuple[Constraint, ...]`

### Objective
- `metric: str` (metric name)
- `target: float | None` (optional)
- `direction: str | None` (recommended: `"min"` or `"max"`)
- `weight: float` (default 1.0)

### Constraint
- `metric: str`
- `lower: float | None`
- `upper: float | None`

**Invariants**
- Objective and constraint `metric` names MUST refer to known metrics (by convention), but validation MAY be delayed until runtime.

## 21.3 CircuitIR

**Purpose:** a normalized, safe-to-transform representation of a SPICE netlist.

**Type:** `eesizer_core.contracts.artifacts.CircuitIR`

**Fields**
- `lines: tuple[str, ...]`  
  Normalized netlist lines.
- `elements: tuple[Element, ...]`  
  Parsed elements with their node lists.
- `param_locs: dict[str, TokenLoc]`  
  Where each editable parameter lives.

### Element
- `name: str`
- `nodes: tuple[str, ...]`

### TokenLoc
- `line_idx: int`
- `token_idx: int`
- `original: str`  
  Original token text.
- `category: str`  
  Free-form label (for example `"value"`, `"w"`, `"l"`).

**Invariants**
- `lines` MUST be the source of truth for reconstruction.
- `param_locs[param_id]` MUST point into `lines`.

## 21.4 Parameter space

See [22_artifact_patch.md](22_artifact_patch.md) for how parameter changes are represented.

`ParamSpace` and `ParamDef` live in `eesizer_core.contracts.artifacts`.

