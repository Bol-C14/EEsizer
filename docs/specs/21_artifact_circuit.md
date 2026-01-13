# 21. Circuit Artifacts

This spec matches the dataclasses in `eesizer_core.contracts.artifacts`.

## 21.1 CircuitSource

**Purpose:** the immutable input source for a circuit (netlist, HDL, etc.).

**Type:** `eesizer_core.contracts.artifacts.CircuitSource`

**Fields**
- `kind: SourceKind`
- `text: str`  
  Raw source text. For SPICE, this is the netlist content.
- `name: str` (default `"circuit"`)
- `metadata: dict[str, Any]`  
  Optional context such as `base_dir` for `.include` resolution or `include_files` for runners.

**Invariants**
- `kind` MUST be correct for the data in `text`.
- For SPICE netlists in EEsizer, `kind` MUST be `SourceKind.spice_netlist`.

**Fingerprint**
- `CircuitSource.fingerprint()` hashes `kind` + `text` (metadata is not part of the identity today).

## 21.2 CircuitSpec

**Purpose:** the declared intent for optimisation (goals + constraints).

**Type:** `eesizer_core.contracts.artifacts.CircuitSpec`

**Fields**
- `objectives: tuple[Objective, ...]`
- `constraints: tuple[Constraint, ...]`
- `observables: tuple[str, ...]`
- `notes: dict[str, Any]`

### Objective
- `metric: str` (metric name)
- `target: float | None` (optional)
- `tol: float | None` (optional tolerance; strategy defines semantics)
- `weight: float` (default 1.0)
- `sense: str` (default `"ge"`, allowed values `"ge"`, `"le"`, `"eq"`)

### Constraint
- `kind: str`
- `data: dict[str, Any]`

**Invariants**
- Objective metric names SHOULD refer to known metrics (validated by strategies).
- Constraint payload is opaque at the contract layer and interpreted by strategies.

## 21.3 CircuitIR

**Purpose:** a normalized, safe-to-transform representation of a SPICE netlist.

**Type:** `eesizer_core.contracts.artifacts.CircuitIR`

**Fields**
- `lines: tuple[str, ...]`  
  Normalized netlist lines.
- `elements: mapping[str, Element]`
- `param_locs: mapping[str, TokenLoc]`
- `includes: tuple[str, ...]`
- `warnings: tuple[str, ...]`

### Element
- `name: str`
- `etype: str`
- `nodes: tuple[str, ...]`
- `model_or_subckt: str | None`
- `params: mapping[str, TokenLoc]`
- `line_idx: int | None`

### TokenLoc
- `line_idx: int`
- `token_idx: int`
- `key: str`                 (for example `"w"`, `"l"`, `"dc"`)
- `raw_token: str`           (for example `"W=1u"`)
- `value_span: tuple[int, int]`  (slice within `raw_token` that holds the value)

**Invariants**
- `lines` MUST be the source of truth for reconstruction.
- `param_locs[param_id]` MUST point into `lines`.
- `includes` MUST be sanitized (no traversal/absolute paths).

## 21.4 Parameter space

See [22_artifact_patch.md](22_artifact_patch.md) for how parameter changes are represented.

`ParamSpace` and `ParamDef` live in `eesizer_core.contracts.artifacts`.
